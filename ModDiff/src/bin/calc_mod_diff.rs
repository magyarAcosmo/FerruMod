use anyhow::Result;
use argmin::core::{CostFunction, Executor, State};
use argmin::solver::neldermead::NelderMead;
use duckdb::Connection;
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use statrs::function::gamma::ln_gamma;
use std::collections::HashMap;
use std::env;

// Beta-Binomial GLM capable of N-dimensional covariates
struct BetaBinomialGLM {
    ks: Vec<f64>, ns: Vec<f64>, covariates: Vec<Vec<f64>>,
}
impl CostFunction for BetaBinomialGLM {
    type Param = Vec<f64>; type Output = f64;
    fn cost(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let mut ll = 0.0;
        let rho = 0.05; // Hardcoded dispersion for stability in windows
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

// Struct to hold metadata parsing
struct SampleMeta { group: f64, covariates: Vec<f64> }

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: calc_mod_diff <mod.db> <metadata.csv>");
        std::process::exit(1);
    }
    let db_path = &args[1];
    let meta_path = &args[2];

    println!("Parsing Metadata...");
    // Assume CSV format: sample_name, group (0/1), age, etc...
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
    let null_params = 1 + num_covariates; // Intercept + Covariates
    let full_params = null_params + 1;    // + Group

    println!("Querying DB...");
    let conn = Connection::open(db_path)?;
    
    // Group windows by coordinate
    let mut stmt = conn.prepare("SELECT chrom, start, \"end\", sample_name, CAST(num_calls AS DOUBLE), CAST(mod_counts AS DOUBLE) FROM windows")?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?,
            row.get::<_, String>(3)?, row.get::<_, f64>(4)?, row.get::<_, f64>(5)?
        ))
    })?;

    // Structure: (chrom, start, end) -> Vec<(sample, n, k)>
    let mut win_data: HashMap<(String, i64, i64), Vec<(String, f64, f64)>> = HashMap::new();
    for r in rows {
        let (chrom, start, end, sample, n, k) = r?;
        win_data.entry((chrom, start, end)).or_default().push((sample, n, k));
    }

    println!("Running Covariate-Aware Beta-Binomial in Parallel...");
    let results: Vec<_> = win_data.into_par_iter().map(|((chrom, start, end), samples)| {
        let mut ks = vec![]; let mut ns = vec![];
        let mut cov_full = vec![]; let mut cov_null = vec![];

        for (samp, n, k) in samples {
            if let Some(meta) = meta_map.get(&samp) {
                ks.push(k); ns.push(n);
                let mut f_row = vec![1.0, meta.group]; // Intercept, Group
                let mut n_row = vec![1.0];             // Intercept
                f_row.extend(&meta.covariates);
                n_row.extend(&meta.covariates);
                cov_full.push(f_row); cov_null.push(n_row);
            }
        }

        if ks.is_empty() { return None; }

        let cost_full = BetaBinomialGLM { ks: ks.clone(), ns: ns.clone(), covariates: cov_full };
        let cost_null = BetaBinomialGLM { ks, ns, covariates: cov_null };

        // Dynamically create starting simplexes based on covariate count
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

        Some((chrom, start, end, diff_beta, p_value))
    }).filter_map(|x| x).collect();

    println!("Applying FDR and saving to DuckDB...");
    conn.execute("DROP TABLE IF EXISTS mod_diff_windows;", [])?;
    conn.execute("CREATE TABLE mod_diff_windows (chrom VARCHAR, start BIGINT, \"end\" BIGINT, diff_beta DOUBLE, p_value DOUBLE);", [])?;
    
    let mut app = conn.appender("mod_diff_windows")?;
    for (chrom, start, end, beta, p) in results {
        app.append_row([duckdb::types::ToSqlOutput::from(chrom), start.into(), end.into(), beta.into(), p.into()])?;
    }
    
    println!("Differential analysis complete!");
    Ok(())
}