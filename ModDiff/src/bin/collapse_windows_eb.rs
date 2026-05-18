use anyhow::Result;
use duckdb::Connection;
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: collapse_windows_eb <my_data.mod.db>");
        std::process::exit(1);
    }
    let db_path = &args[1];

    println!("Connecting to database...");
    let conn = Connection::open(db_path)?;
    
    // Resource Pragmas
    let temp_dir = format!("{}.tmp", db_path);
    let pragma_query = format!("PRAGMA threads=8; PRAGMA temp_directory='{}';", temp_dir);
    conn.execute_batch(&pragma_query)?;

    println!("Step 1: Identifying and grouping adjacent significant windows (FDR < 0.05)...");
    
    // Group windows within 1000bp that have the same direction of effect (hyper vs hypo)
    let merge_sql = r#"
        CREATE TEMP TABLE temp_merged AS
        WITH sig_wins AS (
            SELECT chrom, start, "end", diff_beta, p_value, adj_p_value, shrunk_rho 
            FROM mod_diff_windows_eb 
            WHERE adj_p_value < 0.05 
        ),
        numbered AS (
            SELECT *,
            CASE 
                WHEN LAG("end") OVER w IS NULL 
                  OR LAG("end") OVER w + 1000 < start 
                  OR SIGN(diff_beta) != SIGN(LAG(diff_beta) OVER w) 
                THEN 1 
                ELSE 0 
            END AS new_reg
            FROM sig_wins WINDOW w AS (PARTITION BY chrom ORDER BY start)
        ),
        groups AS (
            SELECT *, SUM(new_reg) OVER (PARTITION BY chrom ORDER BY start ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS reg_id
            FROM numbered
        )
        SELECT 
            chrom, 
            MIN(start) AS start, 
            MAX("end") AS "end", 
            COUNT(*) as win_count,
            
            -- Calculate the average effect size across the merged windows
            AVG(diff_beta) AS mean_diff_beta,
            
            -- The Conservative "Weakest Link" Metric (Region Score)
            MAX(p_value) AS max_p_value,
            MAX(adj_p_value) AS max_adj_p_value,
            
            -- The "Summit" Metric (Biological Peak / Primer Target)
            MIN(adj_p_value) AS peak_adj_p_value,
            
            AVG(shrunk_rho) AS mean_shrunk_rho
        FROM groups 
        GROUP BY chrom, reg_id;
    "#;
    conn.execute(merge_sql, [])?;

    println!("Step 2: Saving collapsed Empirical Bayes DMRs to final table...");
    conn.execute("DROP TABLE IF EXISTS collapsed_dmrs_eb;", [])?;
    
    // Create the final table and map the column names cleanly
    conn.execute(r#"
        CREATE TABLE collapsed_dmrs_eb AS 
        SELECT 
            chrom, 
            start, 
            "end", 
            win_count, 
            mean_diff_beta AS diff_beta, 
            max_p_value AS p_value, 
            max_adj_p_value AS adj_p_value, 
            peak_adj_p_value,
            mean_shrunk_rho AS shrunk_rho
        FROM temp_merged
        ORDER BY adj_p_value ASC, ABS(diff_beta) DESC;
    "#, [])?;
    
    println!("DMRs collapsed successfully! Your mathematically conservative results are sorted in 'collapsed_dmrs_eb'.");
    Ok(())
}