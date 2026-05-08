use anyhow::Result;
use duckdb::Connection;
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: summarize_windows <my_data.mod.db> [window_size] [step_size]");
        std::process::exit(1);
    }
    let db_path = &args[1];
    // Window and step are now at index 2 and 3!
    let window_size: i64 = args.get(2).unwrap_or(&"1000".to_string()).parse()?;
    let step_size: i64 = args.get(3).unwrap_or(&"100".to_string()).parse()?;

    println!("Connecting to {}...", db_path);
    let conn = Connection::open(db_path)?;

    // Enable heavy multi-threading in DuckDB
    conn.execute_batch("PRAGMA threads=8; PRAGMA memory_limit='16GB';")?;

    println!("Step 1: Fast-aggregating raw reads into base-level counts from the 'calls' table...");
    
    // We strictly pull from 'calls' here without needing it as a variable
    let agg_sql = r#"
        CREATE TEMP TABLE temp_positions AS
        SELECT 
            sample_name, 
            chrom, 
            CAST(start AS BIGINT) AS start, 
            COUNT(*) AS num_calls, 
            SUM(CASE WHEN call_code IN ('m', 'h') THEN 1 ELSE 0 END) AS mod_counts
        FROM calls
        WHERE start IS NOT NULL
        GROUP BY sample_name, chrom, start;
    "#;
    conn.execute(agg_sql, [])?;

    println!("Step 2: Building staggered sliding windows (Window: {}bp, Step: {}bp)...", window_size, step_size);
    conn.execute("DROP TABLE IF EXISTS windows;", [])?;

    let query = format!(r#"
        CREATE TABLE windows AS
        WITH RECURSIVE offset_cte AS (
            SELECT 1 AS win_offset
            UNION ALL
            SELECT win_offset + {step} FROM offset_cte WHERE win_offset + {step} < {window}
        ),
        window_map AS (
            SELECT 
                p.sample_name, p.chrom, p.start, 
                p.num_calls, p.mod_counts,
                (p.start - ((p.start - o.win_offset) % {window})) AS win_start
            FROM temp_positions p
            CROSS JOIN offset_cte o
        )
        SELECT 
            sample_name, chrom, 
            win_start AS start, 
            win_start + {window} - 1 AS "end",
            COUNT(*) AS num_CpGs,
            SUM(num_calls) AS num_calls,
            SUM(mod_counts) AS mod_counts
        FROM window_map
        GROUP BY sample_name, chrom, win_start
        HAVING SUM(num_calls) > 0;
    "#, step = step_size, window = window_size);

    conn.execute(&query, [])?;
    println!("Successfully created 'windows' table in the database!");
    
    Ok(())
}