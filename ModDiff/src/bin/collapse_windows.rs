use anyhow::Result;
use argmin::core::{CostFunction, Executor, State};
use argmin::solver::neldermead::NelderMead;
use duckdb::Connection;
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use statrs::function::gamma::ln_gamma;
use std::collections::HashMap;
use std::env;

struct BetaBinomialGLM { ks: Vec<f64>, ns: Vec<f64>, covariates: Vec<Vec<f64>> }
impl CostFunction for BetaBinomialGLM {
    type Param = Vec<f64>; type Output = f64;
    fn cost(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let mut ll = 0.0;
        let rho = 0.05; 
        for i in 0..self.ks.len() {
            let (k, n) = (self.ks[i], self.ns[i]);
            let mut xb = 0.0;
            for j in 0..p.len() { xb += self.covariates[i][j] * p[j]; }
            let mut pi = xb.exp() / (1.0 + xb.exp());
            pi = pi.clamp(1e-5, 1.0 - 1e-5);
            let a = pi * (1.0 - rho) / rho; let b = (1.0 - pi) * (1.0 - rho) / rho;
            ll += ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
                + ln_gamma(k + a) + ln_gamma(n - k + b) - ln_gamma(n + a + b)
                + ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
        }
        Ok(-ll)
    }
}

struct SampleMeta { group: f64, covariates: Vec<f64> }

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: collapse_windows <my_data.mod.db> <metadata.csv>");
        std::process::exit(1);
    }
    let db_path = &args[1];
    let meta_path = &args[2];

    println!("Parsing Metadata...");
    let mut rdr = csv::Reader::from_path(meta_path)?;
    let mut meta_map: HashMap<String, SampleMeta> = HashMap::new();
    for result in rdr.records() {
        let record = result?;
        let sample = record[0].to_string();
        let group: f64 = record[1].parse().unwrap();
        let mut covs = vec![];
        for i in 2..record.len() { covs.push(record[i].parse().unwrap()); }
        meta_map.insert(sample, SampleMeta { group, covariates: covs });
    }
    let num_covariates = meta_map.values().next().unwrap().covariates.len();
    let null_params = 1 + num_covariates; let full_params = null_params + 1;    

    let conn = Connection::open(db_path)?;
    let temp_dir = format!("{}.tmp", db_path);
    let pragma_query = format!("PRAGMA threads=8; PRAGMA temp_directory='{}';", temp_dir);
    conn.execute_batch(&pragma_query)?;

    println!("Collapsing windows (FDR < 0.05) and pooling counts...");
    // Strict FDR thresholding to establish boundaries
    let merge_sql = r#"
        CREATE TEMP TABLE temp_merged AS
        WITH sig_wins AS (
            SELECT chrom, start, "end", diff_beta 
            FROM mod_diff_windows 
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

    println!("Querying pooled read counts for final stats...");
    let mut stmt = conn.prepare(r#"
        SELECT m.chrom, m.start, m."end", m.win_count, w.sample_name, CAST(SUM(w.num_calls) AS DOUBLE), CAST(SUM(w.mod_counts) AS DOUBLE)
        FROM temp_merged m
        JOIN windows w ON m.chrom = w.chrom AND w.start >= m.start AND w."end" <= m."end"
        GROUP BY m.chrom, m.start, m."end", m.win_count, w.sample_name
    "#)?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?, row.get::<_, i64>(3)?,
            row.get::<_, String>(4)?, row.get::<_, f64>(5)?, row.get::<_, f64>(6)?))
    })?;

    let mut win_data: HashMap<(String, i64, i64, i64), Vec<(String, f64, f64)>> = HashMap::new();
    for r in rows {
        let (chrom, start, end, win_count, sample, n, k) = r?;
        win_data.entry((chrom, start, end, win_count)).or_default().push((sample, n, k));
    }

    println!("Running Regional Beta-Binomial in Parallel...");
    let raw_results: Vec<_> = win_data.into_par_iter().map(|((chrom, start, end, win_count), samples)| {
        let capacity = samples.len();
        let mut ks = Vec::with_capacity(capacity); let mut ns = Vec::with_capacity(capacity);
        let mut cov_full = Vec::with_capacity(capacity); let mut cov_null = Vec::with_capacity(capacity);

        for (samp, n, k) in samples {
            if let Some(meta) = meta_map.get(&samp) {
                ks.push(k); ns.push(n);
                let mut f_row = vec![1.0, meta.group]; let mut n_row = vec![1.0];             
                f_row.extend(&meta.covariates); n_row.extend(&meta.covariates);
                cov_full.push(f_row); cov_null.push(n_row);
            }
        }
        if ks.is_empty() { return None; }

        let cost_full = BetaBinomialGLM { ks: ks.clone(), ns: ns.clone(), covariates: cov_full };
        let cost_null = BetaBinomialGLM { ks, ns, covariates: cov_null };

        let mut simplex_full = vec![vec![0.0; full_params]; full_params + 1];
        for i in 0..=full_params { if i > 0 { simplex_full[i][i-1] = 0.1; } }
        let mut simplex_null = vec![vec![0.0; null_params]; null_params + 1];
        for i in 0..=null_params { if i > 0 { simplex_null[i][i-1] = 0.1; } }

        let s_full = NelderMead::new(simplex_full).with_sd_tolerance(1e-4).unwrap();
        let s_null = NelderMead::new(simplex_null).with_sd_tolerance(1e-4).unwrap();
        let r_full = Executor::new(cost_full, s_full).configure(|state| state.max_iters(100)).run();
        let r_null = Executor::new(cost_null, s_null).configure(|state| state.max_iters(100)).run();

        let mut diff_beta = 0.0_f64; let mut ll_full = 0.0_f64; let mut ll_null = 0.0_f64;
        if let Ok(opt) = r_full { ll_full = -opt.state().get_best_cost(); diff_beta = opt.state().get_best_param().unwrap()[1]; }
        if let Ok(opt) = r_null { ll_null = -opt.state().get_best_cost(); }

        let lr_stat = (2.0 * (ll_full - ll_null)).max(1e-10_f64);
        let p_value = 1.0 - ChiSquared::new(1.0).unwrap().cdf(lr_stat);

        Some((chrom, start, end, win_count, diff_beta, p_value))
    }).filter_map(|x| x).collect();

    println!("Applying Benjamini-Hochberg to Final DMRs...");
    let mut results_with_idx: Vec<(usize, _)> = raw_results.into_iter().enumerate().collect();
    results_with_idx.sort_by(|a, b| b.1.5.partial_cmp(&a.1.5).unwrap_or(std::cmp::Ordering::Equal));
    
    let n = results_with_idx.len() as f64;
    let mut min_adj_p = 1.0;
    let mut final_results = vec![None; results_with_idx.len()];

    for (i, (orig_idx, data)) in results_with_idx.into_iter().enumerate() {
        let rank = n - (i as f64);
        let raw_p = data.5;
        let mut adj_p = raw_p * (n / rank);
        if adj_p > 1.0 { adj_p = 1.0; }
        if adj_p < min_adj_p { min_adj_p = adj_p; } else { adj_p = min_adj_p; }
        final_results[orig_idx] = Some((data.0, data.1, data.2, data.3, data.4, raw_p, adj_p));
    }
    let results: Vec<_> = final_results.into_iter().map(|x| x.unwrap()).collect();

    println!("Saving to DuckDB...");
    conn.execute("DROP TABLE IF EXISTS collapsed_dmrs;", [])?;
    conn.execute("CREATE TABLE collapsed_dmrs (chrom VARCHAR, start BIGINT, \"end\" BIGINT, win_count BIGINT, diff_beta DOUBLE, p_value DOUBLE, adj_p_value DOUBLE);", [])?;
    
    let mut app = conn.appender("collapsed_dmrs")?;
    for (chrom, start, end, win_count, beta, p, adj_p) in results {
        app.append_row([duckdb::types::ToSqlOutput::from(chrom), start.into(), end.into(), win_count.into(), beta.into(), p.into(), adj_p.into()])?;
    }
    
    println!("DMRs collapsed and fully scored! Your results are in 'collapsed_dmrs'.");
    Ok(())
}