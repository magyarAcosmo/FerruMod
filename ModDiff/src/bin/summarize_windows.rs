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
    let window_size: i64 = args.get(2).unwrap_or(&"1000".to_string()).parse()?;
    let step_size: i64 = args.get(3).unwrap_or(&"10".to_string()).parse()?;

    println!("Connecting to {}...", db_path);
    let conn = Connection::open(db_path)?;

    // Enable heavy multi-threading in DuckDB
    conn.execute_batch("PRAGMA threads=8; PRAGMA memory_limit='16GB';")?;

    // Drop table if exists
    conn.execute("DROP TABLE IF EXISTS windows;", [])?;

    println!("Aggregating sliding windows (Window: {}bp, Step: {}bp)...", window_size, step_size);

    // This dynamically creates the staggered sliding windows purely in SQL
    // We assume the input table is 'positions' with columns: sample_name, chrom, start, num_calls, mod_counts
    let query = format!(r#"
        CREATE TABLE windows AS
        WITH RECURSIVE offsets AS (
            SELECT 1 AS offset
            UNION ALL
            SELECT offset + {step} FROM offsets WHERE offset + {step} < {window}
        ),
        window_map AS (
            SELECT 
                p.sample_name, p.chrom, p.start, p.num_calls, p.mod_counts,
                (p.start - ((p.start - o.offset) % {window})) AS win_start
            FROM positions p
            CROSS JOIN offsets o
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