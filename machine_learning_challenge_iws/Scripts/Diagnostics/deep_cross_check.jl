using CSV, DataFrames

println("================================================================================")
println("🔍 REAL DATA CROSS-CHECK - FINDING WHERE DATA ACTUALLY IS")
println("================================================================================")

println("\n📁 LOADING ALL FILES...")

train = CSV.read(joinpath(@__DIR__, "../../Data/train.csv"), DataFrame)
features_train = CSV.read(joinpath(@__DIR__, "../../Data/features_train.csv"), DataFrame)
features_train_aligned = CSV.read(joinpath(@__DIR__, "../../Data/features_train_CORRECTLY_ALIGNED.csv"), DataFrame)





println("✅ Files loaded")

println("\n================================================================================")
println("🎯 STEP 1: UNDERSTANDING YOUR CLAIM")
println("================================================================================")
println("You suspect: Data that SHOULD be in column 2 might actually be in column 3")
println("Or: We're comparing WRONG columns thinking they're the right ones")
println("Let's FIND where the real matching data is!\n")

println("================================================================================")
println("🔍 STEP 2: CHECK ALL COLUMNS IN train.csv FOR PATTERNS")
println("================================================================================")

println("train.csv has these columns: $(names(train))")
println("\nLet's examine EACH column in train.csv:")

for (i, col) in enumerate(names(train))
    println("\n📊 Column $i: '$col'")
    println("   Type: $(eltype(train[!, col]))")
    println("   First 5 values: $(train[1:5, col])")
    println("   Unique values: $(length(unique(train[!, col])))")
    
    # Check if this looks like an ID column
    if occursin("id", lowercase(col)) || eltype(train[!, col]) == String
        println("   📍 This looks like an ID column")
    end
    
    # Check if this looks like a label column
    if occursin("ground", lowercase(col)) || occursin("truth", lowercase(col)) || 
       occursin("label", lowercase(col)) || occursin("target", lowercase(col))
        println("   🏷️  This looks like a label/target column")
    end
end

println("\n================================================================================")
println("🔍 STEP 3: CHECK ALL COLUMNS IN features_train.csv FOR PATTERNS")
println("================================================================================")

println("features_train.csv has $(ncol(features_train)) columns")
println("Let's examine the FIRST 10 columns:")

for i in 1:min(10, ncol(features_train))
    col = names(features_train)[i]
    println("\n📊 Column $i: '$col'")
    println("   Type: $(eltype(features_train[!, col]))")
    println("   First 3 values: $(features_train[1:3, col])")
    
    # Special checks for first column
    if i == 1
        println("   🔍 First column - could be ID column")
        # Check if values look like IDs
        first_val = features_train[1, col]
        if typeof(first_val) <: Integer
            println("   📍 Integer values - likely IDs")
        elseif typeof(first_val) <: AbstractFloat
            println("   🔢 Float values - likely features")
        end
    else
        # Check if values look like features
        first_val = features_train[1, col]
        if typeof(first_val) <: AbstractFloat
            println("   📈 Float values between 0-1 - likely features")
        end
    end
end

println("\n================================================================================")
println("🔍 STEP 4: THE CRITICAL TEST - IS COLUMN 1 IN features_train ACTUALLY COLUMN 2?")
println("================================================================================")

println("\n🎯 YOUR CLAIM: Maybe column 1 in features_train.csv is actually column 2 data")
println("   Or maybe the headers are shifted by one column!")
println("   Let's test this theory...\n")

# Test 1: Check if column 1 values look like feature values (0-1 range)
col1 = names(features_train)[1]
col1_values = features_train[!, col1]

println("TEST 1: Does column 1 ('$col1') look like feature values?")
println("  First 5 values: $(col1_values[1:5])")
if eltype(col1_values) <: AbstractFloat
    min_val = minimum(col1_values)
    max_val = maximum(col1_values)
    println("  Range: $min_val to $max_val")
    if 0 <= min_val <= 1 && 0 <= max_val <= 1
        println("  ✅ YES! These look like feature values (0-1 range)")
    else
        println("  ❌ NO - not in 0-1 range")
    end
else
    println("  ❌ NO - not float values")
end

# Test 2: Check if column 2 values look like IDs
if ncol(features_train) >= 2
    col2 = names(features_train)[2]
    col2_values = features_train[!, col2]
    
    println("\nTEST 2: Does column 2 ('$col2') look like IDs?")
    println("  First 5 values: $(col2_values[1:5])")
    if eltype(col2_values) <: Integer
        println("  ✅ YES! These look like integer IDs")
    elseif eltype(col2_values) <: AbstractFloat
        # Check if they're actually integers stored as floats
        all_integers = all(x -> isinteger(x), col2_values[1:10])
        if all_integers
            println("  ⚠️  Float values that are all integers - could be IDs")
        else
            println("  ❌ NO - float values that aren't integers")
        end
    else
        println("  ❌ NO - not integer values")
    end
end

println("\n================================================================================")
println("🔍 STEP 5: MANUAL DATA MATCHING - FIND WHERE train.csv IDs APPEAR")
println("================================================================================")

# Get train IDs (clean them)
train_ids = [parse(Int, replace(id, ".png" => "")) for id in train.id]
println("train.csv IDs (first 5): $(train_ids[1:5])")

# Search for these IDs in EVERY column of features_train
println("\n🔍 Searching for train IDs in EACH column of features_train.csv...")






global found_matches = false
for (i, col) in enumerate(names(features_train))
    col_values = features_train[!, col]
    
    # Try to convert to same type as train_ids
    try
        # Check if any values match train_ids
        matches = 0
        for train_id in train_ids[1:10]  # Check first 10 train IDs
            if train_id in col_values
                matches += 1
            end
        end
        
        if matches > 0
            println("🎉 COLUMN $i ('$col'): Found $matches/10 train IDs!")
            println("   Example: train ID $(train_ids[1]) found in this column? $(train_ids[1] in col_values)")
           global found_matches = true
            
            # Show where it appears
            idx = findfirst(x -> x == train_ids[1], col_values)
            if idx !== nothing
                println("   Found at row $idx in features_train.csv")
            end
        end
    catch e
        # Column type doesn't match - skip
    end
end

if !found_matches
    println("❌ NO matches found for train IDs in any column of features_train.csv")
end

println("\n================================================================================")
println("🔍 STEP 6: CHECK features_train_CORRECTLY_ALIGNED.csv")
println("================================================================================")

println("Checking features_train_CORRECTLY_ALIGNED.csv...")
col1_aligned = names(features_train_aligned)[1]
col1_aligned_values = features_train_aligned[!, col1_aligned]

println("First column ('$col1_aligned') first 5 values: $(col1_aligned_values[1:5])")

# Check if these match train IDs
global matches_aligned = 0
for train_id in train_ids[1:5]
    if train_id in col1_aligned_values
        matches_aligned += 1
    end
end

println("Matches with train IDs: $matches_aligned/5")











if matches_aligned == 5
    println("✅ PERFECT MATCH! features_train_CORRECTLY_ALIGNED.csv has correct IDs in column 1")
else
    println("❌ MISMATCH! Let's check other columns...")
end

println("\n================================================================================")
println("🔍 STEP 7: CHECK IF HEADERS ARE SHIFTED")
println("================================================================================")

println("Theory: What if the CSV headers are shifted by 1 column?")
println("        Column header 'image_id' might actually be over feature data")
println("        And the REAL IDs start from column 2\n")

# Let's check what happens if we treat column 2 as IDs
if ncol(features_train) >= 2
    potential_id_col = features_train[!, names(features_train)[2]]
    println("Column 2 values (first 5): $(potential_id_col[1:5])")
    
    # Check if these could be IDs
    if eltype(potential_id_col) <: Integer
        println("These ARE integers - could be IDs!")
        
        # Check if they match train IDs
        potential_matches = 0
        for train_id in train_ids[1:5]
            if train_id in potential_id_col
                potential_matches += 1
            end
        end
        println("Matches with train IDs: $potential_matches/5")
    end
end

println("\n================================================================================")
println("🎯 STEP 8: LET'S READ THE RAW CSV TO SEE HEADERS")
println("================================================================================")

# Read first few lines of raw CSV
println("Reading raw CSV data (first 3 lines):")

# Read as raw text
try
    lines = readlines("features_train.csv")[1:3]
    for (i, line) in enumerate(lines)
        println("Line $i: $(line[1:min(100, length(line))])...")
    end
catch e
    println("Could not read raw file: $e")
end

println("\n================================================================================")
println("🔍 STEP 9: TEST DIFFERENT READING OPTIONS")
println("================================================================================")

println("Trying different ways to read the CSV...")

# Try reading without header
println("\n1. Reading WITHOUT header (treating first row as data):")
try
    df_noheader = CSV.read("Data/features_train.csv", DataFrame; header=false)
    println("   First row as data: $(df_noheader[1, 1:5])")
    println("   Second row as data: $(df_noheader[2, 1:5])")
    
    # Check if first row looks like headers or data
    first_row = df_noheader[1, :]
    println("\n   First row values (as potential headers):")
    for i in 1:5
        val = first_row[i]
        println("   Column $i: '$val' (type: $(typeof(val)))")
    end
catch e
    println("   Error: $e")
end

println("\n================================================================================")
println("🏆 FINAL DIAGNOSIS")
println("================================================================================")

println("\nBased on ALL checks above:\n")

# Summarize findings
println("1. train.csv:")
println("   - Column 1: 'id' (string IDs with .png)")
println("   - Column 2: 'ground_truth' (0/1 labels)")

println("\n2. features_train.csv:")
println("   - Column 1: 'image_id' (values: $(features_train[1:3, 1]))")
println("   - Are these REALLY IDs or feature data?")

println("\n3. Key question: Is 'image_id' column ACTUALLY:")
println("   a) Real image IDs (should match train.csv)?")
println("   b) Feature values mislabeled as IDs?")
println("   c) Something else entirely?")

println("\n🚨 To answer this, I need YOU to check:")
println("   - Open features_train.csv in a text editor")
println("   - Look at the FIRST FEW ROWS")
println("   - Is column 1 REALLY '100028, 100047, 100050' etc?")
println("   - Or is there a header issue?")

println("\n📋 QUICK CHECK YOU CAN DO:")
println("   head -n 3 features_train.csv")
println("   This will show the actual raw data!")

println("\n================================================================================")
println("🔍 CROSS-CHECK COMPLETE")
println("================================================================================")
