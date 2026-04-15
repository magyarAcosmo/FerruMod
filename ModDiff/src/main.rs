use anyhow::Result;
use argmin::core::{CostFunction, Executor, State};
use argmin::solver::neldermead::NelderMead;
use polars::prelude::*;
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use statrs::function::gamma::ln_gamma;
use std::collections::HashMap;
use std::env;
use std::fs::File;

// --- CONFIGURATION PARSER ---
struct Config {
    mode: String,
    window_size: i64,
    step_size: i64,
    collapse_thresh: f64, // Raw p-value to trigger window merging (Defaults to 0.05)
    p_thresh: f64,        // Adjusted p-value to filter final output CSV
    control_dir: String,
    treatment_dir: String,
    output_path: String,
}

impl Config {
    fn parse() -> Self {
        let args: Vec<String> = env::args().collect();
        let mut config = Config {
            mode: "dmp".to_string(),
            window_size: 100,
            step_size: 50,
            collapse_thresh: 0.05, // Hardcoded default for DMR collapsing
            p_thresh: 1.0,         // Defaults to 1.0 (keep everything) unless specified
            control_dir: String::new(),
            treatment_dir: String::new(),
            output_path: String::new(),
        };

        let mut i = 1;
        let mut positional_args = Vec::new();

        while i < args.len() {
            match args[i].as_str() {
                "--dmp" => config.mode = "dmp".to_string(),
                "--dmr" => config.mode = "dmr".to_string(),
                "--window" => { config.window_size = args[i + 1].parse().unwrap(); i += 1; }
                "--step" => { config.step_size = args[i + 1].parse().unwrap(); i += 1; }
                "--collapse-thresh" => { config.collapse_thresh = args[i + 1].parse().unwrap(); i += 1; } // NEW
                "--p-thresh" => { config.p_thresh = args[i + 1].parse().unwrap(); i += 1; } // NOW CONTROLS FINAL PADJ
                _ => positional_args.push(args[i].clone()),
            }
            i += 1;
        }

        if positional_args.len() != 3 {
            eprintln!("Usage: ferrumod_stats [--dmp | --dmr] [--window 100] [--step 50] [--collapse-thresh 0.05] [--p-thresh 0.05] <control_dir> <treat_dir> <out_base.csv>");
            std::process::exit(1);
        }

        config.control_dir = positional_args[0].trim_end_matches('/').to_string();
        config.treatment_dir = positional_args[1].trim_end_matches('/').to_string();
        config.output_path = positional_args[2].clone();
        config
    }
}

// --- MATH ENGINE ---
struct BetaBinomialGLM {
    ks: Vec<f64>, ns: Vec<f64>, covariates: Vec<Vec<f64>>,
}

impl CostFunction for BetaBinomialGLM {
    type Param = Vec<f64>; type Output = f64;
    fn cost(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let mut log_likelihood = 0.0;
        let rho = 0.05;

        for i in 0..self.ks.len() {
            let k = self.ks[i]; let n = self.ns[i];
            let mut xb = 0.0;
            for j in 0..p.len() { xb += self.covariates[i][j] * p[j]; }
            
            let pi = xb.exp() / (1.0 + xb.exp());
            let pi = pi.clamp(1e-5, 1.0 - 1e-5);
            let a = pi * (1.0 - rho) / rho; let b = (1.0 - pi) * (1.0 - rho) / rho;

            let ll = ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
                + ln_gamma(k + a) + ln_gamma(n - k + b) - ln_gamma(n + a + b)
                + ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
            log_likelihood += ll;
        }
        Ok(-log_likelihood)
    }
}

#[derive(Clone, Debug)]
struct StatResult {
    chrom: String, start: i64, end: i64, mod_type: i32,
    m_frac_ctrl: f64, m_frac_treat: f64,
    n_ctrl: f64, n_treat: f64,
    diff_beta: f64, p_value: f64,
}

fn solve_glm(chrom: &str, start: i64, end: i64, mod_type: i32, k_ctrl: f64, n_ctrl: f64, k_treat: f64, n_treat: f64) -> StatResult {
    let m_frac_ctrl = if n_ctrl > 0.0 { k_ctrl / n_ctrl } else { 0.0 };
    let m_frac_treat = if n_treat > 0.0 { k_treat / n_treat } else { 0.0 };

    if n_ctrl == 0.0 || n_treat == 0.0 {
        return StatResult { chrom: chrom.to_string(), start, end, mod_type, m_frac_ctrl, m_frac_treat, n_ctrl, n_treat, diff_beta: 0.0, p_value: 1.0 };
    }

    let ks = vec![k_ctrl, k_treat]; let ns = vec![n_ctrl, n_treat];
    let covariates_full = vec![vec![1.0, 0.0], vec![1.0, 1.0]];
    let covariates_null = vec![vec![1.0], vec![1.0]];

    let cost_full = BetaBinomialGLM { ks: ks.clone(), ns: ns.clone(), covariates: covariates_full };
    let cost_null = BetaBinomialGLM { ks, ns, covariates: covariates_null };

    let solver_full = NelderMead::new(vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.1]]).with_sd_tolerance(1e-4).unwrap();
    let solver_null = NelderMead::new(vec![vec![0.0], vec![0.1]]).with_sd_tolerance(1e-4).unwrap();

    let res_full = Executor::new(cost_full, solver_full).configure(|state| state.max_iters(100)).run();
    let res_null = Executor::new(cost_null, solver_null).configure(|state| state.max_iters(100)).run();

    let mut diff_beta = 0.0; let mut ll_full = 0.0; let mut ll_null = 0.0;
    if let Ok(opt) = res_full { ll_full = -opt.state().get_best_cost(); if let Some(p) = opt.state().get_best_param() { diff_beta = p[1]; } }
    if let Ok(opt) = res_null { ll_null = -opt.state().get_best_cost(); }

    let mut lr_stat = 2.0 * (ll_full - ll_null);
    if lr_stat < 1e-10 { lr_stat = 1e-10; } 
    let p_value = 1.0 - ChiSquared::new(1.0).unwrap().cdf(lr_stat);

    StatResult { chrom: chrom.to_string(), start, end, mod_type, m_frac_ctrl, m_frac_treat, n_ctrl, n_treat, diff_beta, p_value }
}

fn apply_bh(results: &mut Vec<StatResult>) -> Vec<f64> {
    if results.is_empty() { return vec![]; }
    let m = results.len() as f64;
    let mut pvals: Vec<(usize, f64)> = results.iter().enumerate().map(|(i, r)| (i, r.p_value)).collect();
    pvals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut padjs = vec![0.0; results.len()];
    let mut min_padj: f64 = 1.0;
    for (rank_minus_1, &(idx, p)) in pvals.iter().enumerate().rev() {
        let padj = (p * m / (rank_minus_1 + 1) as f64).min(1.0);
        min_padj = min_padj.min(padj);
        padjs[idx] = min_padj;
    }
    padjs
}

// --- PURE RUST BASE STRUCTURE ---
#[derive(Clone)]
struct BaseStat { pos: i64, k_ctrl: f64, n_ctrl: f64, k_treat: f64, n_treat: f64 }

fn main() -> Result<()> {
    let config = Config::parse();
    println!("Running FerruMod Stats in [{}] mode.", config.mode.to_uppercase());

    let control_glob = format!("{}/*.parquet", config.control_dir);
    let treat_glob = format!("{}/*.parquet", config.treatment_dir);

    // 1. Polars Aggregation: Collapse to base-level stats
    let lf_control = LazyFrame::scan_parquet(&control_glob, Default::default())?.with_column(lit(0.0).alias("group_id"));
    let lf_treat = LazyFrame::scan_parquet(&treat_glob, Default::default())?.with_column(lit(1.0).alias("group_id"));
    let combined = concat(vec![lf_control, lf_treat], UnionArgs::default())?;

    let aggregated = combined
        .explode([col("mod_offsets"), col("mod_probs"), col("mod_type")]) 
        .with_columns([
            (col("pos") + col("mod_offsets").cast(DataType::Int64)).alias("abs_pos"),
            (col("mod_probs").cast(DataType::Float64) / lit(255.0)).alias("mod_weight")
        ])
        .group_by([col("chrom"), col("abs_pos"), col("mod_type"), col("group_id")]) 
        .agg([ col("mod_weight").sum().alias("k"), col("mod_weight").count().alias("n") ])
        .collect()?;

    // 2. Transfer Data to Rust HashMap for fast querying and interval building
    let mut data_map: HashMap<(String, i32), Vec<BaseStat>> = HashMap::new();
    
    let chroms: Vec<String> = aggregated.column("chrom")?.str()?.into_no_null_iter().map(|s| s.to_string()).collect();
    let positions: Vec<i64> = aggregated.column("abs_pos")?.i64()?.into_no_null_iter().collect();
    let mod_types: Vec<i32> = aggregated.column("mod_type")?.i32()?.into_no_null_iter().collect();
    let groups: Vec<f64> = aggregated.column("group_id")?.f64()?.into_no_null_iter().collect();
    let ks: Vec<f64> = aggregated.column("k")?.f64()?.into_no_null_iter().collect();
    let ns: Vec<u32> = aggregated.column("n")?.u32()?.into_no_null_iter().collect();

    for i in 0..aggregated.height() {
        let key = (chroms[i].clone(), mod_types[i]);
        let stat = BaseStat {
            pos: positions[i],
            k_ctrl: if groups[i] == 0.0 { ks[i] } else { 0.0 }, n_ctrl: if groups[i] == 0.0 { ns[i] as f64 } else { 0.0 },
            k_treat: if groups[i] == 1.0 { ks[i] } else { 0.0 }, n_treat: if groups[i] == 1.0 { ns[i] as f64 } else { 0.0 },
        };
        
        let entry = data_map.entry(key).or_insert_with(Vec::new);
        if let Some(existing) = entry.iter_mut().find(|b| b.pos == stat.pos) {
            existing.k_ctrl += stat.k_ctrl; existing.n_ctrl += stat.n_ctrl;
            existing.k_treat += stat.k_treat; existing.n_treat += stat.n_treat;
        } else {
            entry.push(stat);
        }
    }

    for bases in data_map.values_mut() { bases.sort_by_key(|b| b.pos); }

    let mut all_results: Vec<StatResult> = Vec::new();

    if config.mode == "dmp" {
        println!("Calculating Single-Base DMPs...");
        for ((chrom, mod_type), bases) in data_map.into_iter() {
            let mut mod_results: Vec<StatResult> = bases.into_par_iter().map(|b| {
                solve_glm(&chrom, b.pos, b.pos, mod_type, b.k_ctrl, b.n_ctrl, b.k_treat, b.n_treat)
            }).collect();
            all_results.append(&mut mod_results);
        }
    } else if config.mode == "dmr" {
        println!("Calculating DMRs (Window: {}bp, Step: {}bp)...", config.window_size, config.step_size);
        
        for ((chrom, mod_type), bases) in data_map.into_iter() {
            if bases.is_empty() { continue; }
            let max_pos = bases.last().unwrap().pos;
            
            let mut windows = Vec::new();
            let mut w_start = 0;
            while w_start <= max_pos {
                windows.push((w_start, w_start + config.window_size));
                w_start += config.step_size;
            }

            let window_results: Vec<StatResult> = windows.into_par_iter().filter_map(|(start, end)| {
                let mut k_c = 0.0; let mut n_c = 0.0; let mut k_t = 0.0; let mut n_t = 0.0;
                for b in bases.iter().filter(|b| b.pos >= start && b.pos < end) {
                    k_c += b.k_ctrl; n_c += b.n_ctrl; k_t += b.k_treat; n_t += b.n_treat;
                }
                if n_c > 0.0 && n_t > 0.0 {
                    Some(solve_glm(&chrom, start, end, mod_type, k_c, n_c, k_t, n_t))
                } else { None }
            }).collect();

            // Merge windows based on the new `collapse_thresh`
            let mut sig_windows: Vec<StatResult> = window_results.into_iter().filter(|w| w.p_value < config.collapse_thresh).collect();
            sig_windows.sort_by_key(|w| w.start);

            let mut merged_intervals: Vec<(i64, i64)> = Vec::new();
            for w in sig_windows {
                if let Some(last) = merged_intervals.last_mut() {
                    if w.start <= last.1 { last.1 = last.1.max(w.end); } 
                    else { merged_intervals.push((w.start, w.end)); }
                } else {
                    merged_intervals.push((w.start, w.end));
                }
            }

            let mut final_dmrs: Vec<StatResult> = merged_intervals.into_par_iter().map(|(start, end)| {
                let mut k_c = 0.0; let mut n_c = 0.0; let mut k_t = 0.0; let mut n_t = 0.0;
                for b in bases.iter().filter(|b| b.pos >= start && b.pos < end) {
                    k_c += b.k_ctrl; n_c += b.n_ctrl; k_t += b.k_treat; n_t += b.n_treat;
                }
                solve_glm(&chrom, start, end, mod_type, k_c, n_c, k_t, n_t)
            }).collect();

            all_results.append(&mut final_dmrs);
        }
    }

    // --- WRITE FINAL CSVS ---
    let base_output_name = config.output_path.trim_end_matches(".csv");
    let mut unique_mods: Vec<i32> = all_results.iter().map(|r| r.mod_type).collect();
    unique_mods.sort_unstable(); unique_mods.dedup();

    for m_type in unique_mods {
        let mod_letter = match m_type { 0 => "a", 1 => "m", 2 => "h", _ => "unknown" };
        let mut type_results: Vec<StatResult> = all_results.iter().filter(|r| r.mod_type == m_type).cloned().collect();
        
        let all_padjs = apply_bh(&mut type_results);

        // Filter based on the new `p_thresh` controlling final padj
        let mut filtered_results = Vec::new();
        let mut filtered_padjs = Vec::new();

        for (i, r) in type_results.into_iter().enumerate() {
            if all_padjs[i] <= config.p_thresh {
                filtered_results.push(r);
                filtered_padjs.push(all_padjs[i]);
            }
        }

        println!("Modification '{}': Kept {} / {} significant regions/positions.", mod_letter, filtered_results.len(), all_padjs.len());

        if filtered_results.is_empty() {
            println!("Skipping CSV creation for {} (No significant results below --p-thresh).", mod_letter);
            continue;
        }

        let out_chroms = Series::new("chrom", filtered_results.iter().map(|r| r.chrom.as_str()).collect::<Vec<_>>());
        let out_starts = Series::new(if config.mode == "dmp" { "pos" } else { "start" }, filtered_results.iter().map(|r| r.start).collect::<Vec<_>>());
        let mut cols = vec![out_chroms, out_starts];
        
        if config.mode == "dmr" {
            cols.push(Series::new("end", filtered_results.iter().map(|r| r.end).collect::<Vec<_>>()));
        }

        cols.push(Series::new("mod_type", filtered_results.iter().map(|_| mod_letter).collect::<Vec<_>>()));
        cols.push(Series::new(&format!("{}_frac_ctrl", mod_letter), filtered_results.iter().map(|r| r.m_frac_ctrl).collect::<Vec<_>>()));
        cols.push(Series::new(&format!("{}_frac_treat", mod_letter), filtered_results.iter().map(|r| r.m_frac_treat).collect::<Vec<_>>()));
        cols.push(Series::new("coverage_ctrl", filtered_results.iter().map(|r| r.n_ctrl as u32).collect::<Vec<_>>()));
        cols.push(Series::new("coverage_treat", filtered_results.iter().map(|r| r.n_treat as u32).collect::<Vec<_>>()));
        cols.push(Series::new("diff_beta", filtered_results.iter().map(|r| r.diff_beta).collect::<Vec<_>>()));
        cols.push(Series::new("p_value", filtered_results.iter().map(|r| r.p_value).collect::<Vec<_>>()));
        cols.push(Series::new("padj", filtered_padjs));

        let mut out_df = DataFrame::new(cols)?;
        let type_output_path = format!("{}_{}_{}.csv", base_output_name, config.mode, mod_letter);
        CsvWriter::new(&mut File::create(&type_output_path)?).finish(&mut out_df)?;
        println!("Saved: {}", type_output_path);
    }
    Ok(())
}