use anyhow::Result;
use argmin::core::{CostFunction, Executor, State};
use argmin::solver::neldermead::NelderMead;
use polars::prelude::*;
use rayon::prelude::*;
use statrs::function::gamma::ln_gamma;
use std::fs::File;

/// The mathematical model for the Beta-Binomial GLM
struct BetaBinomialGLM {
    ks: Vec<f64>,              // Weighted modification successes (sum of mod_probs / 255)
    ns: Vec<f64>,              // Total coverage (trials)
    covariates: Vec<Vec<f64>>, // e.g., [ [1.0, 0.0], [1.0, 1.0] ] (Intercept + Group)
}

impl CostFunction for BetaBinomialGLM {
    type Param = Vec<f64>; // Our Betas (e.g., [Intercept, Treatment_Effect])
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let mut log_likelihood = 0.0;
        let rho = 0.05; // Fixed overdispersion factor for this example

        for i in 0..self.ks.len() {
            let k = self.ks[i];
            let n = self.ns[i];
            
            // 1. Logit Link Function: Calculate linear predictor (Xb)
            let mut xb = 0.0;
            for j in 0..p.len() {
                xb += self.covariates[i][j] * p[j];
            }
            
            // 2. Inverse Logit to get Probability (pi)
            let pi = xb.exp() / (1.0 + xb.exp());
            let pi = pi.clamp(1e-5, 1.0 - 1e-5); // Prevent math domain errors at extremes

            // 3. Shape parameters for the Beta distribution
            let a = pi * (1.0 - rho) / rho;
            let b = (1.0 - pi) * (1.0 - rho) / rho;

            // 4. Beta-Binomial Log-Likelihood (using Gamma for fractional 'k')
            let ll = ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
                + ln_gamma(k + a) + ln_gamma(n - k + b) - ln_gamma(n + a + b)
                + ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
            
            log_likelihood += ll;
        }
        
        // Argmin MINIMIZES the cost, so we return the negative log-likelihood
        Ok(-log_likelihood)
    }
}

/// Runs the optimization for a single genomic coordinate
fn solve_differential_site(ks: Vec<f64>, ns: Vec<f64>, groups: Vec<f64>) -> f64 {
    // Covariates: Intercept (1.0) and Group ID (0.0 or 1.0)
    let covariates: Vec<Vec<f64>> = groups.iter().map(|&g| vec![1.0, g]).collect();

    let cost_func = BetaBinomialGLM { ks, ns, covariates };

    let initial_simplex = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.0],
        vec![0.0, 0.1],
    ];

    let solver = NelderMead::new(initial_simplex)
        .with_sd_tolerance(1e-4).unwrap();

    let res = Executor::new(cost_func, solver)
        .configure(|state| state.max_iters(100))
        .run();

    // FIX: Safer unwrapping to satisfy type inference
    match res {
        Ok(opt_res) => {
            if let Some(best_params) = opt_res.state().get_best_param() {
                best_params[1] // Return the GroupEffect Beta (gamma_1)
            } else {
                0.0
            }
        },
        Err(_) => 0.0, 
    }
}

fn main() -> Result<()> {
    // --- DYNAMIC DIRECTORY PARSING ---
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <control_dir> <treatment_dir> <output_results.csv>", args[0]);
        std::process::exit(1);
    }
    
    // Trim trailing slashes just in case the user types "data/" instead of "data"
    let control_dir = args[1].trim_end_matches('/');
    let treatment_dir = args[2].trim_end_matches('/');
    let output_path = &args[3];

    // Create the wildcard glob patterns
    let control_glob = format!("{}/*.parquet", control_dir);
    let treat_glob = format!("{}/*.parquet", treatment_dir);
    // --------------------------------

    println!("Scanning Control directory: {}", control_glob);
    println!("Scanning Treatment directory: {}", treat_glob);
    
    // 1. Load Data using Globs! Polars will lazily merge all matching files.
    let lf_control = LazyFrame::scan_parquet(&control_glob, Default::default())?
        .with_column(lit(0.0).alias("group_id"));
        
    let lf_treat = LazyFrame::scan_parquet(&treat_glob, Default::default())?
        .with_column(lit(1.0).alias("group_id"));

    // 2. Concatenate all files from both groups
    let combined = concat(
        vec![lf_control, lf_treat],
        UnionArgs::default()
    )?;

    println!("Aggregating coverage and modifications per genomic site...");

    // 3. ETL Data Prep
    let aggregated = combined
        .explode([col("mod_offsets"), col("mod_probs")])
        .with_columns([
            (col("pos") + col("mod_offsets").cast(DataType::Int64)).alias("abs_pos"),
            (col("mod_probs").cast(DataType::Float64) / lit(255.0)).alias("mod_weight")
        ])
        // Step 3a: Group by Chrom, Pos, and Group to get the successes and coverage
        .group_by([col("chrom"), col("abs_pos"), col("group_id")])
        .agg([
            col("mod_weight").sum().alias("k_successes"),
            col("mod_weight").count().alias("n_coverage") // Works perfectly because threshold is 0
        ])
        // Step 3b: Group again by Chrom and Pos to collect groups into lists for the solver
        .group_by([col("chrom"), col("abs_pos")])
        .agg([
            col("k_successes").alias("ks"),
            col("n_coverage").alias("ns"),
            col("group_id").alias("groups")
        ])
        .collect()?;

    println!("Data prepped! Found {} unique coordinates.", aggregated.height());
    println!("Running Beta-Binomial solver across all sites in parallel...");

    // Extract columns to standard Rust vectors for Rayon parallelization
    let chroms: Vec<String> = aggregated.column("chrom")?.str()?.into_no_null_iter().map(|s| s.to_string()).collect();
    let positions: Vec<i64> = aggregated.column("abs_pos")?.i64()?.into_no_null_iter().collect();
    
    // Polars List chunks to Rust Vec<f64>
    let ks_series = aggregated.column("ks")?.list()?;
    let ns_series = aggregated.column("ns")?.list()?;
    let groups_series = aggregated.column("groups")?.list()?;

    // 4. Run the Solver in Parallel over every site
    let results: Vec<(String, i64, f64)> = (0..aggregated.height()).into_par_iter().map(|i| {
        // Extract the sub-lists for this specific row
        let s_ks = ks_series.get_as_series(i).unwrap();
        let s_ns = ns_series.get_as_series(i).unwrap();
        let s_groups = groups_series.get_as_series(i).unwrap();

        let row_ks: Vec<f64> = s_ks.f64().unwrap().into_no_null_iter().collect();
        let row_ns: Vec<f64> = s_ns.u32().unwrap().into_no_null_iter().map(|x| x as f64).collect();
        let row_groups: Vec<f64> = s_groups.f64().unwrap().into_no_null_iter().collect();

        // Calculate the Differential Modification Beta
        let diff_beta = solve_differential_site(row_ks, row_ns, row_groups);

        (chroms[i].clone(), positions[i], diff_beta)
    }).collect();

    println!("Calculations complete. Writing output to {}", output_path);

    // 5. Build Final Output DataFrame
    let out_chroms = Series::new("chrom", results.iter().map(|r| r.0.as_str()).collect::<Vec<_>>());
    let out_pos = Series::new("pos", results.iter().map(|r| r.1).collect::<Vec<_>>());
    let out_betas = Series::new("diff_beta", results.iter().map(|r| r.2).collect::<Vec<_>>());

    let mut out_df = DataFrame::new(vec![out_chroms, out_pos, out_betas])?;

    // Write to CSV
    let mut file = File::create(output_path)?;
    CsvWriter::new(&mut file).finish(&mut out_df)?;

    println!("Pipeline finished successfully!");
    Ok(())
}