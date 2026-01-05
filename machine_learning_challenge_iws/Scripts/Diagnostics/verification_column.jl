using CSV, DataFrames
using Statistics

# Get the project root directory
const SCRIPT_DIR = @__DIR__
const PROJECT_ROOT = joinpath(SCRIPT_DIR, "..", "..")
const DATA_DIR = joinpath(PROJECT_ROOT, "Data")

println("================================================================================")
println("🔍 SMART COLUMN-AWARE DATA VERIFICATION (FIXED WITH TRUTH)")
println("Script directory: $SCRIPT_DIR")
println("Project root: $PROJECT_ROOT")
println("Data directory: $DATA_DIR")
println("================================================================================\n")

println("📊 STEP 1: SMART COLUMN ANALYSIS (WITH SAFE RANGES)")

println("\n📁 CRITICAL FILES ANALYSIS:")

function analyze_file(filename, max_rows=5, max_cols=5)
    if !isfile(filename)
        println("  ❌ File not found: $filename")
        return nothing
    end
    
    try
        df = CSV.read(filename, DataFrame)
        println("\nAnalyzing: $filename")
        println("  Columns: $(names(df))")
        println("  Total Rows: $(nrow(df))")
        
        # Show first few rows
        println("  First $(min(max_rows, nrow(df))) rows (first $(min(max_cols, ncol(df))) columns):")
        for i in 1:min(max_rows, nrow(df))
            row_str = "    Row $i: "
            for j in 1:min(max_cols, ncol(df))
                val = df[i, j]
                col_name = names(df)[j]
                row_str *= "$(col_name): $val"
                if j < min(max_cols, ncol(df))
                    row_str *= ", "
                end
            end
            println(row_str)
        end
        return df
    catch e
        println("  ❌ Error reading $filename: $e")
        return nothing
    end
end

# Analyze files - USE ABSOLUTE PATHS
train_path = joinpath(DATA_DIR, "train.csv")
test_path = joinpath(DATA_DIR, "test.csv")
features_train_path = joinpath(DATA_DIR, "features_train.csv")
features_test_path = joinpath(DATA_DIR, "features_test.csv")
features_train_correct_path = joinpath(DATA_DIR, "features_train_CORRECTLY_ALIGNED.csv")
features_test_correct_path = joinpath(DATA_DIR, "features_test_CORRECTLY_ALIGNED.csv")
submission_path = joinpath(DATA_DIR, "final_submission.csv")

train_df = analyze_file(train_path)
test_df = analyze_file(test_path)
features_train_df = analyze_file(features_train_path)
features_test_df = analyze_file(features_test_path)
features_train_correct_df = analyze_file(features_train_correct_path)
features_test_correct_df = analyze_file(features_test_correct_path)
submission_df = analyze_file(submission_path)

println("\n================================================================================")
println("🔄 STEP 2: TRUTHFUL CROSS-COLUMN COMPARISON")
println("================================================================================\n")

function compare_ids_truthfully(df1, df2, name1, name2, id_col1, id_col2)
    println("🔗 Comparing $name1 ↔ $name2:")
    
    # Clean IDs
    if eltype(df1[!, id_col1]) == String
        ids1 = [parse(Int, replace(id, ".png" => "")) for id in df1[!, id_col1]]
    else
        ids1 = df1[!, id_col1]
    end
    
    if eltype(df2[!, id_col2]) == String
        ids2 = [parse(Int, replace(id, ".png" => "")) for id in df2[!, id_col2]]
    else
        ids2 = df2[!, id_col2]
    end
    
    println("  $name1 IDs (first 5): $(ids1[1:min(5, length(ids1))])")
    println("  $name2 IDs (first 5): $(ids2[1:min(5, length(ids2))])")
    
    # Calculate REAL overlap
    overlap = intersect(Set(ids1), Set(ids2))
    overlap_count = length(overlap)
    total_count = length(ids1)
    
    if overlap_count == total_count && length(ids1) == length(ids2)
        println("  ✅ EXACT SAME SET OF IDs ($overlap_count/$total_count)")
        
        # Check order
        if ids1 == ids2
            println("  ✅ AND SAME ORDER!")
        else
            println("  ⚠️  DIFFERENT ORDER - needs alignment")
            for i in 1:min(10, length(ids1), length(ids2))
                if ids1[i] != ids2[i]
                    println("  First mismatch at row $i:")
                    println("    $name1: $(ids1[i])")
                    println("    $name2: $(ids2[i])")
                    break
                end
            end
        end
    elseif overlap_count == total_count
        println("  ✅ SAME SET OF IDs ($overlap_count/$total_count)")
        println("  ⚠️  DIFFERENT NUMBER OF ROWS: $(length(ids1)) vs $(length(ids2))")
    elseif overlap_count > 0
        println("  ⚠️  PARTIAL OVERLAP: $overlap_count/$total_count IDs match")
    else
        println("  ❌ COMPLETELY DIFFERENT IDs: 0/$total_count match")
        println("  These are TWO DIFFERENT DATASETS!")
    end
    println()
end

# Compare the important pairs
if train_df !== nothing && features_train_df !== nothing
    compare_ids_truthfully(train_df, features_train_df, "train.csv", "features_train.csv", "id", "image_id")
end

if train_df !== nothing && features_train_correct_df !== nothing
    compare_ids_truthfully(train_df, features_train_correct_df, "train.csv", "features_train_CORRECTLY_ALIGNED.csv", "id", "image_id")
end

if test_df !== nothing && features_test_df !== nothing
    compare_ids_truthfully(test_df, features_test_df, "test.csv", "features_test.csv", "id", "image_id")
end

if test_df !== nothing && features_test_correct_df !== nothing
    compare_ids_truthfully(test_df, features_test_correct_df, "test.csv", "features_test_CORRECTLY_ALIGNED.csv", "id", "image_id")
end

if test_df !== nothing && submission_df !== nothing
    compare_ids_truthfully(test_df, submission_df, "test.csv", "final_submission.csv", "id", "id")
end

println("\n================================================================================")
println("🎯 STEP 3: VERIFYING THE MAPPING MIRACLE (TRUTH VERSION)")
println("================================================================================\n")

if test_df !== nothing && features_test_df !== nothing && features_test_correct_df !== nothing
    println("🔍 THE RE-MAPPING STORY:")
    
    # Get IDs
    test_ids = [parse(Int, replace(id, ".png" => "")) for id in test_df.id]
    features_test_ids = features_test_df.image_id
    features_test_correct_ids = features_test_correct_df.image_id
    
    println("  COMPETITION test.csv IDs (first 5): $(test_ids[1:5])")
    println("  Your local test image IDs (first 5): $(features_test_ids[1:5])")
    println("  Aligned feature file IDs (first 5): $(features_test_correct_ids[1:5])")
    println()
    
    # Check if mapping worked
    if Set(test_ids) == Set(features_test_correct_ids)
        println("  ✅ MAPPING SUCCESSFUL!")
        println("  Your alignment script re-mapped Local IDs → COMPETITION IDs")
        
        # Check order
        if test_ids == features_test_correct_ids
            println("  ✅ AND in CORRECT ORDER for submission!")
        else
            println("  ⚠️  But DIFFERENT ORDER - needs reordering for submission")
        end
    else
        overlap = length(intersect(Set(test_ids), Set(features_test_correct_ids)))
        println("  ❌ MAPPING FAILED: Only $overlap/$(length(test_ids)) IDs match")
    end
end

println("\n================================================================================")
println("📤 STEP 4: SUBMISSION VALIDITY CHECK")
println("================================================================================\n")

if submission_df !== nothing
    println("📊 final_submission.csv analysis:")
    println("  Total predictions: $(nrow(submission_df))")
    println("  Range: $(minimum(submission_df.ground_truth)) to $(maximum(submission_df.ground_truth))")
    println("  Mean: $(mean(submission_df.ground_truth))")
    
    # Check for invalid probabilities
    invalid_below = count(x -> x < 0, submission_df.ground_truth)
    invalid_above = count(x -> x > 1, submission_df.ground_truth)
    total_invalid = invalid_below + invalid_above
    
    if total_invalid > 0
        println("\n  🚨 INVALID PROBABILITIES FOUND!")
        println("  Values < 0: $invalid_below")
        println("  Values > 1: $invalid_above")
        println("  Total invalid: $total_invalid/$(nrow(submission_df))")
        
        # Find examples
        println("\n  Examples of invalid predictions:")
        examples_count = 0  # Changed variable name
        for i in 1:nrow(submission_df)
            val = submission_df.ground_truth[i]
            if val < 0 || val > 1
                id_val = submission_df.id[i]
                println("    Row $i: ID=$id_val, Prediction=$val")
                examples_count += 1
                if examples_count >= 3
                    break
                end
            end
        end
        
        println("\n  💡 FIX: Add clipping to prediction script:")
        println("  predictions = max.(0, min.(1, raw_predictions))")
    else
        println("\n  ✅ ALL predictions in valid range [0,1]")
    end
end

println("\n================================================================================")
println("🏆 FINAL VERDICT (TRUTH EDITION)")
println("================================================================================\n")

println("📋 SUMMARY OF REALITY:")
println("1. train.csv: $(train_df !== nothing ? nrow(train_df) : "NOT FOUND") rows")
println("2. test.csv: $(test_df !== nothing ? nrow(test_df) : "NOT FOUND") rows")
println("3. features_train.csv: $(features_train_df !== nothing ? nrow(features_train_df) : "NOT FOUND") rows")
println("4. features_test_CORRECTLY_ALIGNED.csv: $(features_test_correct_df !== nothing ? nrow(features_test_correct_df) : "NOT FOUND") rows")
println("5. final_submission.csv: $(submission_df !== nothing ? nrow(submission_df) : "NOT FOUND") rows")

println("\n🔍 CRITICAL FINDINGS:")

# Check the REAL overlaps
if train_df !== nothing && features_train_df !== nothing
    train_ids = [parse(Int, replace(id, ".png" => "")) for id in train_df.id]
    feature_ids = features_train_df.image_id
    overlap = length(intersect(Set(train_ids), Set(feature_ids)))
    
    if overlap == 0
        println("❌ train.csv and features_train.csv have ZERO overlap!")
        println("   These are COMPLETELY DIFFERENT DATASETS!")
    end
end

if test_df !== nothing && features_test_correct_df !== nothing
    test_ids = [parse(Int, replace(id, ".png" => "")) for id in test_df.id]
    feature_correct_ids = features_test_correct_df.image_id
    
    if Set(test_ids) == Set(feature_correct_ids)
        println("✅ features_test_CORRECTLY_ALIGNED.csv matches test.csv!")
    end
end

println("\n⚠️  WARNING: Your old verification script was lying to you!")
println("   It said 'SAME SET' when IDs were completely different.")
println("   This script shows the TRUTH about your data.")

if submission_df !== nothing
    invalid_count = count(x -> x < 0 || x > 1, submission_df.ground_truth)
    if invalid_count > 0
        println("\n⚠️  WARNING: $invalid_count invalid probabilities in submission")
        println("   Fix with: predictions = max.(0, min.(1, predictions))")
    end
end

println("\n================================================================================")
println("🔍 TRUTHFUL VERIFICATION COMPLETE")
println("================================================================================")
