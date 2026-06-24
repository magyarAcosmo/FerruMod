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

    // Resource Pragmas
    let temp_dir = format!("{}.tmp", db_path);
    let pragma_query = format!("PRAGMA threads=8; PRAGMA temp_directory='{}';", temp_dir);
    conn.execute_batch(&pragma_query)?;

    // Create the final empty table matching the R script's logic
    println!("Initializing 'windows' table...");
    conn.execute("DROP TABLE IF EXISTS windows;", [])?;
    conn.execute(r#"
        CREATE TABLE windows (
            sample_name VARCHAR,
            chrom VARCHAR,
            start BIGINT,
            "end" BIGINT,
            num_sites BIGINT,
            num_calls BIGINT,
            mod_counts BIGINT
        );
    "#, [])?;

    // Get list of all distinct samples to chunk the processing
    let mut stmt = conn.prepare("SELECT DISTINCT sample_name FROM calls WHERE sample_name IS NOT NULL")?;
    let samples: Vec<String> = stmt.query_map([], |row| row.get(0))?.filter_map(Result::ok).collect();

    println!("Found {} samples. Processing iteratively to save memory...", samples.len());

    // Fix: Start offsets at 0 for 0-based coordinates
    let offsets: Vec<i64> = (0..window_size).step_by(step_size as usize).collect();

    for samp in samples {
        println!("  -> Aggregating base positions for sample: {}", samp);
        let samp_esc = samp.replace("'", "''");

        // Step 1: Base counts for THIS SAMPLE ONLY
        conn.execute("DROP TABLE IF EXISTS temp_positions;", [])?;
        let pos_sql = format!(r#"
            CREATE TEMP TABLE temp_positions AS
            SELECT 
                chrom, 
                CAST(start AS BIGINT) AS start, 
                COUNT(*) AS num_calls, 
                SUM(CASE WHEN call_code IN ('m', 'h') THEN 1 ELSE 0 END) AS mod_counts
            FROM calls
            WHERE sample_name = '{}' AND start IS NOT NULL
            GROUP BY chrom, start;
        "#, samp_esc);
        conn.execute(&pos_sql, [])?;

        // Step 2: Loop through offsets iteratively just like the R script
        for offset in &offsets {
            let win_sql = format!(r#"
                INSERT INTO windows
                WITH window_map AS (
                    SELECT 
                        '{}' AS sample_name,
                        chrom,
                        -- Safe FLOOR math to prevent negative modulo corruption at chr starts
                        {} + CAST(FLOOR((start - {}) / CAST({} AS DOUBLE)) AS BIGINT) * {} AS win_start,
                        num_calls, 
                        mod_counts
                    FROM temp_positions
                )
                SELECT 
                    sample_name, 
                    chrom, 
                    win_start AS start, 
                    -- Fix: Removed the "- 1" to maintain half-open intervals
                    win_start + {} AS "end",
                    COUNT(*) AS num_sites,
                    SUM(num_calls) AS num_calls,
                    SUM(mod_counts) AS mod_counts
                FROM window_map
                WHERE win_start >= 0  -- Fix: changed from 1 to 0
                GROUP BY sample_name, chrom, win_start
                HAVING SUM(num_calls) >= 1;
            "#, samp_esc, offset, offset, window_size, window_size, window_size);
            
            conn.execute(&win_sql, [])?;
        }
    }

    println!("Step 3: Indexing database for high-speed downstream access...");
    conn.execute("CREATE INDEX idx_windows ON windows(chrom, start, \"end\");", [])?;
    
    println!("Successfully built chunked 'windows' table!");
    Ok(())
}