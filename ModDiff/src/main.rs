use anyhow::Result;
use argmin::core::{CostFunction, Executor, State};
use argmin::solver::neldermead::NelderMead;
use polars::prelude::*;
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use statrs::function::gamma::ln_gamma;
use std::fs::File;

struct BetaBinomialGLM {
    ks: Vec<f64>,
    ns: Vec<f64>,
    covariates: Vec<Vec<f64>>,
}

impl CostFunction for BetaBinomialGLM {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let mut log_likelihood = 0.0;
        
        let last_idx = p.len() - 1;
        let rho_logit = p[last_idx];
        
        let mut rho = rho_logit.exp() / (1.0 + rho_logit.exp());
        rho = rho.clamp(1e-4, 0.99);

        for i in 0..self.ks.len() {
            let k = self.ks[i];
            let n = self.ns[i];
            
            let mut xb = 0.0;
            for j in 0..last_idx {
                xb += self.covariates[i][j] * p[j];
            }
            
            let pi = xb.exp() / (1.0 + xb.exp());
            let pi = pi.clamp(1e-5, 1.0 - 1e-5);

            let a = pi * (1.0 - rho) / rho;
            let b = (1.0 - pi) * (1.0 - rho) / rho;

            let ll = ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
                + ln_gamma(k + a) + ln_gamma(n - k + b) - ln_gamma(n + a + b)
                + ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
            
            log_likelihood += ll;
        }
        
        Ok(-log_likelihood)
    }
}

// <-- NEW: Added Clone so we can partition the results later
#[derive(Clone)]
struct SiteResult {
    chrom: String,
    pos: i64,
    mod_type: i32,
    m_frac_ctrl: f64,
    m_frac_treat: f64,
    diff_beta: f64,
    p_value: f64,
}

fn solve_differential_site(chrom: String, pos: i64, mod_type: i32, ks: Vec<f64>, ns: Vec<f64>, groups: Vec<f64>) -> SiteResult {
    let mut k_ctrl = 0.0; let mut n_ctrl = 0.0;
    let mut k_treat = 0.0; let mut n_treat = 0.0;
    
    for i in 0..groups.len() {
        if groups[i] == 0.0 { k_ctrl += ks[i]; n_ctrl += ns[i]; }
        else if groups[i] == 1.0 { k_treat += ks[i]; n_treat += ns[i]; }
    }
    
    let m_frac_ctrl = if n_ctrl > 0.0 { k_ctrl / n_ctrl } else { 0.0 };
    let m_frac_treat = if n_treat > 0.0 { k_treat / n_treat } else { 0.0 };

    if n_ctrl == 0.0 || n_treat == 0.0 {
        return SiteResult { chrom, pos, mod_type, m_frac_ctrl, m_frac_treat, diff_beta: 0.0, p_value: 1.0 };
    }

    let covariates_full: Vec<Vec<f64>> = groups.iter().map(|&g| vec![1.0, g]).collect();
    let covariates_null: Vec<Vec<f64>> = vec![vec![1.0]; groups.len()];

    let cost_full = BetaBinomialGLM { ks: ks.clone(), ns: ns.clone(), covariates: covariates_full };
    let cost_null = BetaBinomialGLM { ks, ns, covariates: covariates_null };

    let initial_simplex_full = vec![
        vec![0.0, 0.0, -3.0], vec![0.1, 0.0, -3.0], vec![0.0, 0.1, -3.0], vec![0.0, 0.0, -2.0],
    ];
    let solver_full = NelderMead::new(initial_simplex_full).with_sd_tolerance(1e-4).unwrap();
    let res_full = Executor::new(cost_full, solver_full).configure(|state| state.max_iters(100)).run();

    let initial_simplex_null = vec![
        vec![0.0, -3.0], vec![0.1, -3.0], vec![0.0, -2.0],
    ];
    let solver_null = NelderMead::new(initial_simplex_null).with_sd_tolerance(1e-4).unwrap();
    let res_null = Executor::new(cost_null, solver_null).configure(|state| state.max_iters(100)).run();

    let mut diff_beta = 0.0;
    let mut ll_full = 0.0;
    let mut ll_null = 0.0;

    if let Ok(opt) = res_full {
        ll_full = -opt.state().get_best_cost();
        if let Some(params) = opt.state().get_best_param() { diff_beta = params[1]; }
    }
    if let Ok(opt) = res_null { ll_null = -opt.state().get_best_cost(); }

    let mut lr_stat = 2.0 * (ll_full - ll_null);
    if lr_stat < 1e-10 { lr_stat = 1e-10; } 
    
    let chi_sq = ChiSquared::new(1.0).unwrap();
    let p_value = 1.0 - chi_sq.cdf(lr_stat);

    SiteResult { chrom, pos, mod_type, m_frac_ctrl, m_frac_treat, diff_beta, p_value }
}

fn apply_benjamini_hochberg(results: &mut Vec<SiteResult>) -> Vec<f64> {
    let m = results.len() as f64;
    let mut pvals: Vec<(usize, f64)> = results.iter().enumerate().map(|(i, r)| (i, r.p_value)).collect();
    pvals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut padjs = vec![0.0; results.len()];
    let mut min_padj: f64 = 1.0;

    for (rank_minus_1, &(idx, p)) in pvals.iter().enumerate().rev() {
        let rank = (rank_minus_1 + 1) as f64;
        let padj = (p * m / rank).min(1.0);
        min_padj = min_padj.min(padj);
        padjs[idx] = min_padj;
    }
    padjs
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <control_dir> <treatment_dir> <output_base_name.csv>", args[0]);
        std::process::exit(1);
    }
    
    let control_dir = args[1].trim_end_matches('/');
    let treatment_dir = args[2].trim_end_matches('/');
    let output_path = &args[3];

    // Strip the .csv extension so we can append _mod_X to it later
    let base_output_name = output_path.trim_end_matches(".csv");

    let control_glob = format!("{}/*.parquet", control_dir);
    let treat_glob = format!("{}/*.parquet", treatment_dir);

    println!("Scanning and aggregating data...");
    
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
        .agg([
            col("mod_weight").sum().alias("k_successes"),
            col("mod_weight").count().alias("n_coverage") 
        ])
        .group_by([col("chrom"), col("abs_pos"), col("mod_type")])
        .agg([
            col("k_successes").alias("ks"),
            col("n_coverage").alias("ns"),
            col("group_id").alias("groups")
        ])
        .collect()?;

    println!("Data prepped! Found {} unique coordinates/mod-types.", aggregated.height());
    println!("Running Likelihood Ratio Tests across all sites in parallel...");

    let chroms: Vec<String> = aggregated.column("chrom")?.str()?.into_no_null_iter().map(|s| s.to_string()).collect();
    let positions: Vec<i64> = aggregated.column("abs_pos")?.i64()?.into_no_null_iter().collect();
    let mod_types: Vec<i32> = aggregated.column("mod_type")?.i32()?.into_no_null_iter().collect();
    
    let ks_series = aggregated.column("ks")?.list()?;
    let ns_series = aggregated.column("ns")?.list()?;
    let groups_series = aggregated.column("groups")?.list()?;

    let results: Vec<SiteResult> = (0..aggregated.height()).into_par_iter().map(|i| {
        let s_ks = ks_series.get_as_series(i).unwrap();
        let s_ns = ns_series.get_as_series(i).unwrap();
        let s_groups = groups_series.get_as_series(i).unwrap();

        let row_ks: Vec<f64> = s_ks.f64().unwrap().into_no_null_iter().collect();
        let row_ns: Vec<f64> = s_ns.u32().unwrap().into_no_null_iter().map(|x| x as f64).collect();
        let row_groups: Vec<f64> = s_groups.f64().unwrap().into_no_null_iter().collect();

        solve_differential_site(chroms[i].clone(), positions[i], mod_types[i], row_ks, row_ns, row_groups)
    }).collect();

    // <-- NEW: Find all unique modification types present in the data
    let mut unique_mods = mod_types.clone();
    unique_mods.sort_unstable();
    unique_mods.dedup();

    println!("Calculations complete. Found {} distinct modification types.", unique_mods.len());

    // <-- NEW: Loop through each unique modification type, partition, and write
    for m_type in unique_mods {
        println!("Processing FDR and writing CSV for Modification Type: {}...", m_type);
        
        // Filter results just for this specific modification
        let mut type_results: Vec<SiteResult> = results.iter().filter(|r| r.mod_type == m_type).cloned().collect();

        // Apply BH Correction ONLY to this modification type
        let padjs = apply_benjamini_hochberg(&mut type_results);

        let out_chroms = Series::new("chrom", type_results.iter().map(|r| r.chrom.as_str()).collect::<Vec<_>>());
        let out_pos = Series::new("pos", type_results.iter().map(|r| r.pos).collect::<Vec<_>>());
        let out_mod_type = Series::new("mod_type", type_results.iter().map(|r| r.mod_type).collect::<Vec<_>>());
        let out_mf_ctrl = Series::new("m_frac_ctrl", type_results.iter().map(|r| r.m_frac_ctrl).collect::<Vec<_>>());
        let out_mf_treat = Series::new("m_frac_treat", type_results.iter().map(|r| r.m_frac_treat).collect::<Vec<_>>());
        let out_betas = Series::new("diff_beta", type_results.iter().map(|r| r.diff_beta).collect::<Vec<_>>());
        let out_pvals = Series::new("p_value", type_results.iter().map(|r| r.p_value).collect::<Vec<_>>());
        let out_padjs = Series::new("padj", padjs);

        let mut out_df = DataFrame::new(vec![
            out_chroms, out_pos, out_mod_type, out_mf_ctrl, out_mf_treat, out_betas, out_pvals, out_padjs
        ])?;

        let type_output_path = format!("{}_mod_{}.csv", base_output_name, m_type);
        let mut file = File::create(&type_output_path)?;
        CsvWriter::new(&mut file).finish(&mut out_df)?;
        
        println!("Saved: {}", type_output_path);
    }

    println!("Pipeline finished successfully!");
    Ok(())
}