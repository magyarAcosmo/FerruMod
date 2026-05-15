use anyhow::Result;
use argmin::core::{CostFunction, Executor, State};
use argmin::solver::neldermead::NelderMead;
use duckdb::Connection;
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use statrs::function::gamma::ln_gamma;
use std::collections::HashMap;
use std::env;

struct BetaBinomialGLM { ks: Vec<f64>, ns: Vec<f64>, covariates: Vec<Vec<f64>>, rho: f64 }
impl CostFunction for BetaBinomialGLM {
    type Param = Vec<f64>; type Output = f64;
    fn cost(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        for param in p {
            if *param < -15.0 || *param > 15.0 { return Ok(f64::INFINITY); }
        }
        let mut ll = 0.0;
        for i in 0..self.ks.len() {
            let (k, n) = (self.ks[i], self.ns[i]);
            let mut xb = 0.0;
            for j in 0..p.len() { xb += self.covariates[i][j] * p[j]; }
            let mut pi = xb.exp() / (1.0 + xb.exp());
            pi = pi.clamp(1e-5, 1.0 - 1e-5);
            let a = pi * (1.0 - self.rho) / self.rho; let b = (1.0 - pi) * (1.0 - self.rho) / self.rho;
            ll += ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
                + ln_gamma(k + a) + ln_gamma(n - k + b) - ln_gamma(n + a + b)
                + ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
        }
        let mut penalty = 0.0; let penalty_weight = 0.05;
        for param in p { penalty += penalty_weight * param.powi(2); }
        Ok(-ll + penalty)
    }
}

struct SampleMeta { group: f64, covariates: Vec<f64> }

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: collapse_windows_eb <my_data.mod.db> <metadata.csv> [min_depth]");
        std::process::exit(1);
    }
    let db_path = &args[1]; let meta_path = &args[2];
    let min_depth: f64 = args.get(3).unwrap_or(&"0".to_string()).parse().unwrap_or(0.0);

    println!("Parsing Metadata...");
    let mut rdr = csv::Reader::from_path(meta_path)?;
    let mut meta_map: HashMap<String, SampleMeta> = HashMap::new();
    for result in rdr.records() {
        let record = result?;
        let sample = record[0].to_string(); let group: f64 = record[1].parse().unwrap();
        let mut covs = vec![]; for i in 2..record.len() { covs.push(record[i].parse().unwrap()); }
        meta_map.insert(sample, SampleMeta { group, covariates: covs });
    }
    let num_covariates = meta_map.values().next().unwrap().covariates.len();
    let null_params = 1 + num_covariates; let full_params = null_params + 1;    

    let conn = Connection::open(db_path)?;
    let temp_dir = format!("{}.tmp", db_path);
    let pragma_query = format!("PRAGMA threads=8; PRAGMA temp_directory='{}';", temp_dir);
    conn.execute_batch(&pragma_query)?;

    println!("Collapsing EB windows (FDR < 0.05) and pooling counts...");
    let merge_sql = r#"
        CREATE TEMP TABLE temp_merged AS
        WITH sig_wins AS (
            SELECT chrom, start, "end", diff_beta 
            FROM mod_diff_windows_eb 
            WHERE adj_p_value < 0.05 
        ),
        numbered AS (
            SELECT *,
            CASE WHEN LAG("end") OVER w IS NULL OR LAG("end") OVER w + 1000 < start OR SIGN(diff_beta) != SIGN(LAG(diff_beta) OVER w) THEN 1 ELSE 0 END AS new_reg
            FROM sig_wins WINDOW w AS (PARTITION BY chrom ORDER BY start)
        ),
        groups AS (
            SELECT *, SUM(new_reg) OVER (PARTITION BY chrom ORDER BY start ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS reg_id
            FROM numbered
        )
        SELECT chrom, MIN(start) AS start, MAX("end") AS "end", COUNT(*) as win_count
        FROM groups GROUP BY chrom, reg_id;
    "#;
    conn.execute(merge_sql, [])?;

    println!("Querying pooled read counts for Empirical Bayes Shrinkage...");
    let mut stmt = conn.prepare(r#"
        WITH region_calls AS (
            SELECT c.sample_name, c.chrom, c.start as pos, c.call_code, 
                   m.start as m_start, m."end" as m_end, m.win_count
            FROM calls c
            JOIN temp_merged m ON c.chrom = m.chrom AND c.start >= m.start AND c.start <= m."end"
        )
        SELECT chrom, m_start, m_end, win_count, sample_name, 
               CAST(COUNT(*) AS DOUBLE) AS num_calls, 
               CAST(SUM(CASE WHEN call_code IN ('m', 'h') THEN 1 ELSE 0 END) AS DOUBLE) AS mod_counts,
               CAST(COUNT(DISTINCT pos) AS DOUBLE) AS cpgs
        FROM region_calls
        GROUP BY chrom, m_start, m_end, win_count, sample_name
    "#)?;
    
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?, row.get::<_, i64>(3)?,
            row.get::<_, String>(4)?, row.get::<_, f64>(5)?, row.get::<_, f64>(6)?, row.get::<_, f64>(7)?))
    })?;

    let mut win_data: HashMap<(String, i64, i64, i64), Vec<(String, f64, f64, f64)>> = HashMap::new();
    for r in rows {
        let (chrom, start, end, win_count, sample, n, k, cpgs) = r?;
        win_data.entry((chrom, start, end, win_count)).or_default().push((sample, n, k, cpgs));
    }

    println!("Filtering collapsed EB regions by depth and spatial CpGs...");
    win_data.retain(|_, samples| {
        let mut g0_depth = 0.0; let mut g1_depth = 0.0; let mut max_cpgs = 0.0;
        for (samp, n, _, cpgs) in samples {
            if *cpgs > max_cpgs { max_cpgs = *cpgs; }
            if let Some(meta) = meta_map.get(samp) {
                if meta.group == 0.0 { g0_depth += *n; } else { g1_depth += *n; }
            }
        }
        g0_depth >= min_depth.max(1.0) && g1_depth >= min_depth.max(1.0) && max_cpgs >= 5.0
    });

    println!("Pass 1: Estimating raw dispersion across POOLED regions...");
    let mut raw_rhos_map = HashMap::new(); let mut rhos_vec = Vec::with_capacity(win_data.len());
    for ((chrom, start, end, _), samples) in &win_data {
        let s_count = samples.len() as f64; let mut raw_rho = 1e-4; 
        if s_count > 1.0 {
            let (mut total_n, mut total_k) = (0.0, 0.0);
            for (_, n, k, _) in samples { total_n += n; total_k += k; }
            let p_bar = total_k / total_n; let n_bar = total_n / s_count;
            let mut s2 = 0.0;
            for (_, n, k, _) in samples { let p_s = k / n; s2 += (p_s - p_bar).powi(2); }
            s2 /= s_count - 1.0;
            let binom_v = (p_bar * (1.0 - p_bar)) / n_bar;
            if s2 > binom_v && p_bar > 0.0 && p_bar < 1.0 { raw_rho = (s2 - binom_v) / ((p_bar * (1.0 - p_bar)) - binom_v).max(1e-10); }
        }
        raw_rho = raw_rho.clamp(1e-5, 0.99);
        raw_rhos_map.insert((chrom.clone(), *start, *end), raw_rho); rhos_vec.push(raw_rho);
    }

    println!("Pass 2: Calculating EB Global Prior for regions...");
    rhos_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let global_prior = if rhos_vec.is_empty() { 0.05 } else { rhos_vec[rhos_vec.len() / 2] };

    println!("Pass 3: Running Shrunk Beta-Binomial on pooled data...");
    let raw_results: Vec<_> = win_data.into_par_iter().map(|((chrom, start, end, win_count), samples)| {
        let capacity = samples.len(); let s_count = capacity as f64;
        let mut ks = Vec::with_capacity(capacity); let mut ns = Vec::with_capacity(capacity);
        let mut cov_full = Vec::with_capacity(capacity); let mut cov_null = Vec::with_capacity(capacity);

        let (mut total_n, mut total_k) = (0.0, 0.0);
        for (_, n, k, _) in &samples { total_n += n; total_k += k; }
        let mut p_avg = if total_n > 0.0 { total_k / total_n } else { 0.5 };
        p_avg = p_avg.clamp(1e-4, 1.0 - 1e-4);
        let intercept_start = (p_avg / (1.0 - p_avg)).ln();

        for (samp, n, k, _) in samples {
            if let Some(meta) = meta_map.get(&samp) {
                ks.push(k); ns.push(n);
                let mut f_row = vec![1.0, meta.group]; let mut n_row = vec![1.0];             
                f_row.extend(&meta.covariates); n_row.extend(&meta.covariates);
                cov_full.push(f_row); cov_null.push(n_row);
            }
        }
        if ks.is_empty() { return None; }

        let raw_rho = raw_rhos_map.get(&(chrom.clone(), start, end)).unwrap();
        let df_prior = 10.0; let rho_shrunk = ((raw_rho * s_count) + (global_prior * df_prior)) / (s_count + df_prior);

        let cost_full = BetaBinomialGLM { ks: ks.clone(), ns: ns.clone(), covariates: cov_full, rho: rho_shrunk };
        let cost_null = BetaBinomialGLM { ks, ns, covariates: cov_null, rho: rho_shrunk };

        let mut simplex_full = vec![vec![0.0; full_params]; full_params + 1];
        for i in 0..=full_params { simplex_full[i][0] = intercept_start; if i > 0 { simplex_full[i][i-1] = 0.5; } }
        let mut simplex_null = vec![vec![0.0; null_params]; null_params + 1];
        for i in 0..=null_params { simplex_null[i][0] = intercept_start; if i > 0 { simplex_null[i][i-1] = 0.5; } }

        let s_full = NelderMead::new(simplex_full).with_sd_tolerance(1e-6).unwrap();
        let s_null = NelderMead::new(simplex_null).with_sd_tolerance(1e-6).unwrap();
        
        let r_full = Executor::new(cost_full, s_full).configure(|state| state.max_iters(2500)).run();
        let r_null = Executor::new(cost_null, s_null).configure(|state| state.max_iters(2500)).run();

        let mut diff_beta = 0.0_f64; let mut ll_full = 0.0_f64; let mut ll_null = 0.0_f64;
        
        if let Ok(opt) = r_full { 
            let params = opt.state().get_best_param().unwrap();
            let mut penalty = 0.0; for p in params { penalty += 0.05 * (*p).powi(2); }
            ll_full = -(opt.state().get_best_cost() - penalty); 
            
            let intercept = params[0]; let diff_log = params[1];
            let mu_ctrl = intercept.exp() / (1.0 + intercept.exp());
            let case_logit = intercept + diff_log;
            let mu_case = case_logit.exp() / (1.0 + case_logit.exp());
            diff_beta = mu_case - mu_ctrl;
        }
        if let Ok(opt) = r_null { 
            let params = opt.state().get_best_param().unwrap();
            let mut penalty = 0.0; for p in params { penalty += 0.05 * (*p).powi(2); }
            ll_null = -(opt.state().get_best_cost() - penalty); 
        }

        let lr_stat = (2.0 * (ll_full - ll_null)).max(1e-10_f64);
        let p_value = 1.0 - ChiSquared::new(1.0).unwrap().cdf(lr_stat);

        Some((chrom, start, end, win_count, diff_beta, p_value, rho_shrunk))
    }).filter_map(|x| x).collect();

    println!("Applying Benjamini-Hochberg to Final EB Regions...");
    let mut results_with_idx: Vec<(usize, _)> = raw_results.into_iter().enumerate().collect();
    results_with_idx.sort_by(|a, b| b.1.5.partial_cmp(&a.1.5).unwrap_or(std::cmp::Ordering::Equal));
    
    let n = results_with_idx.len() as f64;
    let mut min_adj_p = 1.0; let mut final_results = vec![None; results_with_idx.len()];

    for (i, (orig_idx, data)) in results_with_idx.into_iter().enumerate() {
        let rank = n - (i as f64); let raw_p = data.5;
        let mut adj_p = raw_p * (n / rank);
        if adj_p > 1.0 { adj_p = 1.0; }
        if adj_p < min_adj_p { min_adj_p = adj_p; } else { adj_p = min_adj_p; }
        final_results[orig_idx] = Some((data.0, data.1, data.2, data.3, data.4, raw_p, adj_p, data.6));
    }
    let results: Vec<_> = final_results.into_iter().map(|x| x.unwrap()).collect();

    println!("Saving to DuckDB...");
    conn.execute("DROP TABLE IF EXISTS collapsed_dmrs_eb;", [])?;
    conn.execute("CREATE TABLE collapsed_dmrs_eb (chrom VARCHAR, start BIGINT, \"end\" BIGINT, win_count BIGINT, diff_beta DOUBLE, p_value DOUBLE, adj_p_value DOUBLE, shrunk_rho DOUBLE);", [])?;
    
    let mut app = conn.appender("collapsed_dmrs_eb")?;
    for (chrom, start, end, win_count, beta, p, adj_p, rho) in results {
        app.append_row([duckdb::types::ToSqlOutput::from(chrom), start.into(), end.into(), win_count.into(), beta.into(), p.into(), adj_p.into(), rho.into()])?;
    }
    
    println!("Empirical Bayes DMRs collapsed and scored! Your results are in 'collapsed_dmrs_eb'.");
    Ok(())
}