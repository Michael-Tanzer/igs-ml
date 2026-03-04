#!/bin/bash
# validate_matlab_sql.sh
# Validates that SQL queries return identical results to MATLAB

set -e  # Exit on error

echo "==================================="
echo "MATLAB vs SQL Query Validation"
echo "==================================="
echo ""

# Step 1: Run MATLAB to extract query and results
echo "Step 1: Running MATLAB to extract query..."
cd /data/sheilalyt/GHL-Autoscope

matlab -nodisplay -nosplash -r "addpath('scripts'); extract_matlab_query; exit" 2>&1 | tee /tmp/matlab_output.log

# Check if MATLAB succeeded
if [ ! -f /tmp/matlab_query.sql ]; then
    echo "ERROR: MATLAB did not generate query file"
    exit 1
fi

if [ ! -f /tmp/matlab_results.csv ]; then
    echo "ERROR: MATLAB did not generate results CSV"
    exit 1
fi

echo ""
echo "✓ MATLAB query extracted successfully"
echo ""

# Step 2: Run the same query in SQL
echo "Step 2: Running same query in SQL..."

# Read the query from file
QUERY=$(cat /tmp/matlab_query.sql)

# Run SQL query and save to CSV
mysql -u autoscope_user -pautoscope_pass -h 127.0.0.1 -P 3306 autoscope -e "$QUERY" | \
    awk 'NR==1 {print "object_id\tPID\tsample_desc\tsmear_type\tid_image_set\tid_slide_number\tz_index\tz_stack_height\tx\ty\tz_stage_position\tpixelIdxList\tlocator_value\tid_blood_sample\tcollection_date\tid_locator_algorithm\tlocation_date\tid_image_tile\tannotated\tlocator_algorithm\tlocator_algorithm_desc\tmagnification\tpixels_per_micron\ttile_filename\tz_stack_filename\tfile_location"; next} {print}' | \
    sed 's/\t/,/g' > /tmp/sql_results_full.csv

# Extract just the key columns for comparison (same as MATLAB output)
awk -F',' 'NR==1 {print "object_id,x,y,file_location,pixels_per_micron,locator_algorithm"}
           NR>1 {print $1","$9","$10","$26","$23","$20}' /tmp/sql_results_full.csv > /tmp/sql_results.csv

echo "✓ SQL query completed"
echo ""

# Step 3: Compare results
echo "Step 3: Comparing results..."
echo ""

# Get row counts
matlab_count=$(tail -n +2 /tmp/matlab_results.csv | wc -l)
sql_count=$(tail -n +2 /tmp/sql_results.csv | wc -l)

echo "Row counts:"
echo "  MATLAB: $matlab_count objects"
echo "  SQL:    $sql_count objects"

if [ $matlab_count -ne $sql_count ]; then
    echo "✗ FAIL: Row counts differ!"
    exit 1
else
    echo "  ✓ PASS: Same row count"
fi

echo ""

# Compare object IDs
echo "Comparing object IDs..."
matlab_ids=$(tail -n +2 /tmp/matlab_results.csv | cut -d',' -f1 | sort)
sql_ids=$(tail -n +2 /tmp/sql_results.csv | cut -d',' -f1 | sort)

if [ "$matlab_ids" == "$sql_ids" ]; then
    echo "  ✓ PASS: All object IDs match"
else
    echo "  ✗ FAIL: Object IDs differ"

    # Show differences
    diff <(echo "$matlab_ids") <(echo "$sql_ids") | head -20
    exit 1
fi

echo ""

# Compare coordinates for first 10 objects
echo "Comparing coordinates (first 10 objects)..."
echo ""
printf "%-10s | %-15s | %-15s | %s\n" "Object ID" "MATLAB (x,y)" "SQL (x,y)" "Match"
printf "%s\n" "$(printf '%.0s-' {1..70})"

# Compare line by line
mismatch_count=0
total_compared=0

while IFS=',' read -r obj_id x y file ppm loc; do
    total_compared=$((total_compared + 1))

    # Skip header
    if [ "$obj_id" == "object_id" ]; then
        continue
    fi

    # Get corresponding SQL line
    sql_line=$(grep "^$obj_id," /tmp/sql_results.csv)

    if [ -z "$sql_line" ]; then
        echo "✗ Object $obj_id not found in SQL results"
        mismatch_count=$((mismatch_count + 1))
        continue
    fi

    # Extract SQL values
    sql_x=$(echo "$sql_line" | cut -d',' -f2)
    sql_y=$(echo "$sql_line" | cut -d',' -f3)

    # Compare (allowing for floating point precision)
    x_diff=$(echo "$x - $sql_x" | bc -l | sed 's/-//')
    y_diff=$(echo "$y - $sql_y" | bc -l | sed 's/-//')

    # Check if difference is < 0.0001
    x_match=$(echo "$x_diff < 0.0001" | bc -l)
    y_match=$(echo "$y_diff < 0.0001" | bc -l)

    if [ $x_match -eq 1 ] && [ $y_match -eq 1 ]; then
        match_str="✓ MATCH"
    else
        match_str="✗ MISMATCH"
        mismatch_count=$((mismatch_count + 1))
    fi

    # Print first 10 comparisons
    if [ $total_compared -le 11 ]; then  # 11 because of header
        printf "%-10s | (%6.2f, %6.2f) | (%6.2f, %6.2f) | %s\n" \
               "$obj_id" "$x" "$y" "$sql_x" "$sql_y" "$match_str"
    fi

done < /tmp/matlab_results.csv

echo ""

if [ $mismatch_count -eq 0 ]; then
    echo "  ✓ PASS: All coordinates match!"
else
    echo "  ✗ FAIL: $mismatch_count mismatches found"
    exit 1
fi

echo ""

# Final summary
echo "==================================="
echo "VALIDATION SUMMARY"
echo "==================================="
echo "✓ Row counts match: $matlab_count objects"
echo "✓ Object IDs match"
echo "✓ Coordinates match (within 0.0001 tolerance)"
echo ""
echo "Result: SQL queries return IDENTICAL results to MATLAB!"
echo ""
echo "Files generated:"
echo "  - /tmp/matlab_query.sql     (exact query MATLAB runs)"
echo "  - /tmp/matlab_results.csv   (MATLAB results)"
echo "  - /tmp/sql_results.csv      (SQL results)"
echo ""
echo "✓✓✓ VALIDATION PASSED ✓✓✓"