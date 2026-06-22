#!/bin/bash

DB_PATH=$1 # Path to the DuckDB database file
META_PATH=$2 # Path to the metadata csv file

echo "Testing depth thresholds to find optimal independent filter..."
echo "Depth, Significant_DMRs, Median_Rho" > optimization_results.csv

depths=(3 5 8 10 12 15 20)

for depth in "${depths[@]}"; do
    echo "Running pipeline for min_depth = $depth..."
    
    # Run your exact Rust binary
    ./calc_mod_diff_eb $DB_PATH $META_PATH $depth
    
    # Query DuckDB for the number of hits (FDR < 0.05) and the average dispersion
    HITS=$(duckdb $DB_PATH -csv -c "SELECT COUNT(*) FROM mod_diff_windows_eb WHERE adj_p_value < 0.05;" | tail -n 1)
    AVG_RHO=$(duckdb $DB_PATH -csv -c "SELECT AVG(shrunk_rho) FROM mod_diff_windows_eb;" | tail -n 1)
    
    echo "Depth $depth: $HITS hits (Avg Rho: $AVG_RHO)"
    echo "$depth, $HITS, $AVG_RHO" >> optimization_results.csv
done

echo "Optimization complete! Check optimization_results.csv."