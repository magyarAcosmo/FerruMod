use anyhow::Result;
use duckdb::Connection;
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: collapse_windows <my_data.mod.db>");
        std::process::exit(1);
    }
    let db_path = &args[1];

    let conn = Connection::open(db_path)?;
    println!("Collapsing windows and recalculating true stats...");

    // 1. Merge contiguous significant windows using pure SQL window functions
    let merge_sql = r#"
        CREATE TEMP TABLE temp_merged AS
        WITH sig_wins AS (
            SELECT chrom, start, end, diff_beta 
            FROM mod_diff_windows 
            WHERE p_value < 0.05 
        ),
        numbered AS (
            SELECT *,
            CASE WHEN LAG(end) OVER w IS NULL OR LAG(end) OVER w + 1000 < start OR SIGN(diff_beta) != SIGN(LAG(diff_beta) OVER w) THEN 1 ELSE 0 END AS new_reg
            FROM sig_wins WINDOW w AS (PARTITION BY chrom ORDER BY start)
        ),
        groups AS (
            SELECT *, SUM(new_reg) OVER (PARTITION BY chrom ORDER BY start ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS reg_id
            FROM numbered
        )
        SELECT chrom, MIN(start) AS start, MAX(end) AS end, COUNT(*) as win_count
        FROM groups GROUP BY chrom, reg_id;
    "#;
    conn.execute(merge_sql, [])?;

    // 2. Query back the RAW counts for the new merged boundaries
    // We join the temp_merged bounds against the raw 'windows' table to get the true pooled read counts
    let agg_sql = r#"
        CREATE TABLE collapsed_dmrs AS
        SELECT 
            m.chrom, m.start, m.end, m.win_count,
            w.sample_name, SUM(w.num_calls) AS pooled_n, SUM(w.mod_counts) AS pooled_k
        FROM temp_merged m
        JOIN windows w ON m.chrom = w.chrom AND w.start >= m.start AND w.end <= m.end
        GROUP BY m.chrom, m.start, m.end, m.win_count, w.sample_name;
    "#;
    
    conn.execute("DROP TABLE IF EXISTS collapsed_dmrs;", [])?;
    conn.execute(agg_sql, [])?;

    // From here, you would typically pull the data back into Rust vectors 
    // and run the EXACT same Beta-Binomial function from `calc_mod_diff.rs` 
    // to populate a final `final_dmrs` table with adjusted p-values.
    
    println!("DMRs collapsed! Extracted raw counts for recalculated regions into 'collapsed_dmrs'.");

    Ok(())
}