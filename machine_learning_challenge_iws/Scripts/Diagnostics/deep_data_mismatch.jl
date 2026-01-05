using DataFrames, CSV, Printf

println("="^80)
println("🚨 DEEP DATA MISMATCH INVESTIGATION")
println("="^80)

# ============================================================================
# 1. LOAD ALL CRITICAL FILES
# ============================================================================
println("\n📁 LOADING ALL CRITICAL FILES...")

# Competition official files
train_csv = CSV.read("Data/train.csv", DataFrame)
test_csv = CSV.read("Data/test.csv", DataFrame)

# Feature files (multiple versions)
feature_files = filter(x -> occursin("feature", lowercase(x)) && endswith(x, ".csv"), readdir("Data/"))
println("Found $(length(feature_files)) feature files")

# ============================================================================
# 2. ANALYZE ID FORMATS
# ============================================================================
println("\n🆔 STEP 1: ANALYZING ID FORMATS")

println("📘 train.csv ID format:")
train_ids = train_csv[!, "id"]
println("  First 5: $(train_ids[1:5])")
println("  Sample check: $(train_ids[100])")
println("  Has .png? $(endswith(string(train_ids[1]), ".png"))")

println("\n📗 test.csv ID format:")
test_ids = test_csv[!, "id"]
println("  First 5: $(test_ids[1:5])")
println("  Sample check: $(test_ids[100])")
println("  Has .png? $(endswith(string(test_ids[1]), ".png"))")

# ============================================================================
# 3. CHECK EACH FEATURE FILE FOR MISMATCHES
# ============================================================================
println("\n🔍 STEP 2: CHECKING EACH FEATURE FILE")

for feature_file in sort(feature_files)
    println("\n📄 Analyzing: $feature_file")
    println("-"^50)
    
    try
        # Load feature file
features = CSV.read("Data/" * feature_file, DataFrame)        
        # Find ID column
        id_col = nothing
        possible_id_cols = ["image_id", "id", "ID", "Image_ID"]
        for col in names(features)
            if col in possible_id_cols
                id_col = col
                break
            end
        end
        
        if id_col === nothing
            println("  ❌ No ID column found! Columns: $(names(features))")
            continue
        end
        
        feature_ids = features[!, Symbol(id_col)]
        println("  ID column: $id_col")
        println("  Rows: $(nrow(features))")
        println("  First 5 IDs: $(feature_ids[1:5])")
        
        # Convert to string and remove .png for comparison
        feature_ids_str = string.(feature_ids)
        feature_ids_clean = [replace(id, ".png" => "") for id in feature_ids_str]
        
        # Check if these are TRAINING or TEST IDs
        train_ids_clean = [replace(string(id), ".png" => "") for id in train_ids]
        test_ids_clean = [replace(string(id), ".png" => "") for id in test_ids]
        
        # Calculate overlaps
        overlap_with_train = length(intersect(Set(feature_ids_clean), Set(train_ids_clean)))
        overlap_with_test = length(intersect(Set(feature_ids_clean), Set(test_ids_clean)))
        
        println("  Overlap with train.csv: $overlap_with_train/$(length(train_ids))")
        println("  Overlap with test.csv:  $overlap_with_test/$(length(test_ids))")
        
        # Determine what type of data this is
        if overlap_with_train == length(feature_ids_clean) && overlap_with_train > 0
            println("  ✅ This appears to be TRAINING features")
        elseif overlap_with_test == length(feature_ids_clean) && overlap_with_test > 0
            println("  ✅ This appears to be TEST features")
        elseif overlap_with_train > 0 && overlap_with_test == 0
            println("  ⚠️  MIXED: Has $(overlap_with_train) training IDs but NO test IDs")
        elseif overlap_with_train == 0 && overlap_with_test > 0
            println("  ⚠️  MIXED: Has $(overlap_with_test) test IDs but NO training IDs")
        elseif overlap_with_train == 0 && overlap_with_test == 0
            println("  🚨 CRITICAL: NO OVERLAP with train.csv OR test.csv!")
            println("     These features are from COMPLETELY DIFFERENT IMAGES!")
        end
        
        # Check for duplicates
        unique_count = length(unique(feature_ids_clean))
        if unique_count < length(feature_ids_clean)
            duplicates = length(feature_ids_clean) - unique_count
            println("  ⚠️  Contains $duplicates duplicate IDs!")
        end
        
    catch e
        println("  ❌ Error reading file: $e")
    end
end

# ============================================================================
# 4. CHECK SUBMISSION FILE
# ============================================================================
println("\n📤 STEP 3: CHECKING SUBMISSION FILE")

if isfile("Data/final_submission.csv")
    submission = CSV.read("Data/final_submission.csv", DataFrame)
    println("📄 final_submission.csv:")
    println("  Rows: $(nrow(submission))")
    println("  Columns: $(names(submission))")
    
    # Check ID match with test.csv
    if "id" in names(submission)
        sub_ids = submission[!, "id"]
        sub_ids_clean = [replace(string(id), ".png" => "") for id in sub_ids]
        test_ids_clean = [replace(string(id), ".png" => "") for id in test_csv[!, "id"]]
        
        # Compare
        if sub_ids == test_csv[!, "id"]
            println("  ✅ IDs EXACTLY match test.csv (perfect!)")
        elseif sub_ids_clean == test_ids_clean
            println("  ✅ IDs match test.csv (after removing .png)")
        else
            # Check overlap
            overlap = length(intersect(Set(sub_ids_clean), Set(test_ids_clean)))
            println("  ❌ ID MISMATCH: Overlap = $overlap/$(length(test_ids_clean))")
            
            if overlap == 0
                println("  🚨 ZERO OVERLAP! Submission for WRONG images!")
                println("     Submission IDs: $(sub_ids_clean[1:5])...")
                println("     Test.csv IDs:   $(test_ids_clean[1:5])...")
            end
        end
        
        # Check prediction values
        if "ground_truth" in names(submission)
            preds = submission[!, "ground_truth"]
            println("  Predictions range: $(minimum(preds)) to $(maximum(preds))")
            
            # Check for invalid probabilities
            invalid_low = sum(preds .< 0)
            invalid_high = sum(preds .> 1)
            if invalid_low > 0 || invalid_high > 0
                println("  ⚠️  INVALID probabilities: $invalid_low <0, $invalid_high >1")
            else
                println("  ✅ All probabilities in valid range [0,1]")
            end
        end
    end
else
    println("  ❌ final_submission.csv not found!")
end

# ============================================================================
# 5. CHECK FOLDER IMAGES
# ============================================================================
println("\n🖼️ STEP 4: CHECKING IMAGE FOLDERS")

function check_image_folder(folder_name, expected_ids)
    println("\nChecking $folder_name/ folder:")
    
    if !isdir(folder_name)
        println("  ❌ Folder not found!")
        return 0, 0
    end
    
    # Get all PNG files
    png_files = filter(x -> endswith(x, ".png"), readdir(folder_name, join=false))
    println("  Found $(length(png_files)) PNG files")
    
    if length(png_files) > 0
        println("  First 5: $(png_files[1:5])")
        
        # Compare with expected IDs
        png_set = Set(png_files)
        expected_set = Set(expected_ids)
        
        overlap = length(intersect(png_set, expected_set))
        missing = length(setdiff(expected_set, png_set))
        extra = length(setdiff(png_set, expected_set))
        
        println("  Overlap with CSV: $overlap/$(length(expected_set))")
        println("  Missing from folder: $missing images")
        println("  Extra in folder: $extra images")
        
        if overlap == 0
            println("  🚨 ZERO OVERLAP! Wrong images in folder!")
        end
        
        return length(png_files), overlap
    end
    
    return 0, 0
end

# Check train folder
train_png_count, train_overlap = check_image_folder("Data/train", train_csv[!, "id"])

# Check test folder  
test_png_count, test_overlap = check_image_folder("Data/test", test_csv[!, "id"])

# ============================================================================
# 6. COMPREHENSIVE MISMATCH ANALYSIS
# ============================================================================
println("\n" * "="^80)
println("📈 COMPREHENSIVE MISMATCH ANALYSIS")
println("="^80)

# Load one feature file for detailed analysis
feature_file = "Data/features_train.csv"
if isfile(feature_file)
    features = CSV.read(feature_file, DataFrame)
    feature_ids = string.(features[!, "image_id"])
    feature_ids_clean = [replace(id, ".png" => "") for id in feature_ids]
    
    train_ids_clean = [replace(string(id), ".png" => "") for id in train_csv[!, "id"]]
    test_ids_clean = [replace(string(id), ".png" => "") for id in test_csv[!, "id"]]
    
    println("\n🔗 FEATURE TO CSV COMPARISON (features_train.csv):")
    
    # Compare with train.csv
    train_match = feature_ids_clean == train_ids_clean
    train_overlap = length(intersect(Set(feature_ids_clean), Set(train_ids_clean)))
    
    println("  Match train.csv exactly? $(train_match ? "✅ YES" : "❌ NO")")
    println("  Overlap with train.csv: $train_overlap/$(length(train_ids_clean))")
    
    if !train_match && train_overlap == length(feature_ids_clean)
        println("  ⚠️  Same IDs but DIFFERENT ORDER")
        println("  First mismatch:")
        for i in 1:min(length(feature_ids_clean), length(train_ids_clean))
            if feature_ids_clean[i] != train_ids_clean[i]
                println("    Row $i: features='$(feature_ids_clean[i])', train.csv='$(train_ids_clean[i])'")
                break
            end
        end
    end
    
    # Compare with test.csv
    test_overlap = length(intersect(Set(feature_ids_clean), Set(test_ids_clean)))
    println("  Overlap with test.csv:  $test_overlap/$(length(test_ids_clean))")
    
    # Check if features are actually from test data (common error)
    if test_overlap == length(feature_ids_clean)
        println("  🚨 features_train.csv actually contains TEST data!")
    end
end

# ============================================================================
# 7. ACTIONABLE FINDINGS
# ============================================================================
println("\n" * "="^80)
println("🚨 ACTIONABLE FINDINGS")
println("="^80)

findings = []

# Check 1: Are train folder images correct?
if train_png_count > 0 && train_overlap == 0
    push!(findings, "❌ train/ folder has WRONG images (0 overlap with train.csv)")
end

# Check 2: Are test folder images correct?
if test_png_count > 0 && test_overlap == 0
    push!(findings, "❌ test/ folder has WRONG images (0 overlap with test.csv)")
end

# Check 3: Feature file content
feature_file = "Data/features_train.csv"
if isfile(feature_file)
    features = CSV.read(feature_file, DataFrame)
    feature_ids = string.(features[!, "image_id"])
    feature_ids_clean = [replace(id, ".png" => "") for id in feature_ids]
    train_ids_clean = [replace(string(id), ".png" => "") for id in train_csv[!, "id"]]
    
    overlap = length(intersect(Set(feature_ids_clean), Set(train_ids_clean)))
    if overlap == 0
        push!(findings, "🚨 features_train.csv has ZERO overlap with train.csv")
        push!(findings, "   → You extracted features from WRONG images!")
    elseif overlap < length(train_ids_clean)
        push!(findings, "⚠️  features_train.csv missing $(length(train_ids_clean)-overlap) training images")
    end
end

# Check 4: Submission alignment
if isfile("Data/final_submission.csv")
    submission = CSV.read("final_submission.csv", DataFrame)
    if "id" in names(submission)
        sub_ids = submission[!, "id"]
        if sub_ids != test_csv[!, "id"]
            push!(findings, "❌ Submission IDs don't match test.csv")
            push!(findings, "   → Submission will score ZERO in competition!")
        end
    end
end

# Print findings
if isempty(findings)
    println("✅ No critical issues found!")
else
    println("Found $(length(findings)) critical issues:")
    for (i, finding) in enumerate(findings)
        println("  $i. $finding")
    end
end

# ============================================================================
# 8. RECOMMENDED FIXES
# ============================================================================
println("\n" * "="^80)
println("🔧 RECOMMENDED FIXES")
println("="^80)

if any(occursin.("WRONG images", findings))
    println("🚨 IMMEDIATE ACTION REQUIRED:")
    println()
    println("1. GET CORRECT IMAGES from competition organizers:")
    println("   • Download train.zip with images matching train.csv IDs")
    println("   • Download test.zip with images matching test.csv IDs")
    println()
    println("2. REPLACE current folders:")
    println("   rm -rf train/ test/")
    println("   unzip train.zip -d train/")
    println("   unzip test.zip -d test/")
    println()
    println("3. RE-EXTRACT features:")
    println("   julia extract_features.jl")
    println()
    println("4. RE-TRAIN model:")
    println("   julia train_xgboost.jl")
    println()
    println("5. RE-PREDICT with correct data:")
    println("   julia predict.jl")
end

# Check specific common issue
if isfile("Data/features_train.csv") && isfile("Data/train.csv")
    features = CSV.read("Data/features_train.csv", DataFrame)
    train = CSV.read("Data/train.csv", DataFrame)
    
    feature_ids = string.(features[!, "image_id"])
    train_ids = string.(train[!, "id"])
    
    if Set(feature_ids) == Set(train_ids)
        println("\n✅ IDs match (same set, maybe different order)")
        println("   Run alignment script:")
        println("   julia final_correct_fix.jl")
    elseif length(intersect(Set(feature_ids), Set(train_ids))) == 0
        println("\n🚨 COMPLETELY DIFFERENT IMAGE SETS!")
        println("   You need the ACTUAL competition images!")
    end
end

println("\n" * "="^80)
println("🔍 ANALYSIS COMPLETE")
println("="^80)
println("\nRun this command to see specific ID mismatches:")
println("julia check_feature_ids.jl")
