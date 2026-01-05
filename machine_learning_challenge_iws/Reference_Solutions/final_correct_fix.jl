# final_correct_fix.jl - UPDATED WITH CORRECT PATHS
using DataFrames, CSV, Printf

println("🔧 FINAL CORRECT FIX - Creating properly aligned features")
println("="^60)

# 1. Load test.csv to get the CORRECT order of IDs
println("\n📄 Loading test.csv to get correct ID order...")
test_csv = CSV.read(joinpath(@__DIR__, "../Data/test.csv"), DataFrame)
correct_ids = [replace(id, ".png" => "") for id in test_csv.id]
println("Correct IDs from test.csv: $(length(correct_ids))")
println("First 5: $(correct_ids[1:5])")

# 2. Check if we have features for these IDs in the ORIGINAL features_test.csv
println("\n🔍 Checking original features_test.csv...")
if !isfile(joinpath(@__DIR__, "../Data/features_test.csv"))
    println("❌ features_test.csv not found!")
    println("You need to re-extract features using: python Python_Scripts/export_transformer.py")
    return
end

orig_features = CSV.read(joinpath(@__DIR__, "../Data/features_test.csv"), DataFrame)
println("Original features: $(nrow(orig_features)) rows")
println("First 5 IDs in original: $(orig_features[1:5, "image_id"])")

# 3. Create mapping from test.csv IDs to feature positions
println("\n🎯 Creating alignment mapping...")
# Check which IDs from test.csv exist in features
orig_ids_set = Set(string.(orig_features[!, "image_id"]))
correct_ids_set = Set(correct_ids)
common_ids = intersect(orig_ids_set, correct_ids_set)
missing_in_features = setdiff(correct_ids_set, orig_ids_set)
extra_in_features = setdiff(orig_ids_set, correct_ids_set)

println("Analysis:")
println(" Total in test.csv: $(length(correct_ids_set))")
println(" Total in features: $(length(orig_ids_set))")
println(" Common IDs: $(length(common_ids))")
println(" Missing in features: $(length(missing_in_features))")
println(" Extra in features: $(length(extra_in_features))")

if length(common_ids) == 0
    println("\n❌ CRITICAL ERROR: NO COMMON IDs between features and test.csv!")
    println(" First 5 feature IDs: $(collect(orig_ids_set)[1:5])")
    println(" First 5 test.csv IDs: $(collect(correct_ids_set)[1:5])")
    println("\nThis means features were extracted from COMPLETELY DIFFERENT images!")
    return
end

if length(missing_in_features) > 0
    println("\n⚠️ WARNING: Missing features for some test.csv IDs!")
    println("First 5 missing: $(collect(missing_in_features)[1:min(5, end)])")
end

# 4. Create correctly aligned features - SIMPLER APPROACH
println("\n💾 Creating correctly aligned features_test.csv...")
# Get feature column names
feature_cols = [col for col in names(orig_features) if col != "image_id"]
num_features = length(feature_cols)
println("Number of features: $num_features")

# Convert features to matrix for faster access
println("Converting features to matrix...")
feature_matrix = Matrix{Float32}(orig_features[!, feature_cols])
id_vector = string.(orig_features[!, "image_id"])

# Create a dictionary mapping ID to row index
println("Creating ID to row index mapping...")
id_to_index = Dict{String, Int}()
for (i, id) in enumerate(id_vector)
    id_to_index[id] = i
end

# Create aligned features in test.csv order
println("Building aligned features...")
missing_count = 0
# Initialize aligned matrix
aligned_matrix = zeros(Float32, length(correct_ids), num_features)
aligned_ids = Vector{String}()

for (i, id) in enumerate(correct_ids)
    id_str = string(id)
    push!(aligned_ids, id_str)
   
    if id_str in keys(id_to_index)
        row_idx = id_to_index[id_str]
        aligned_matrix[i, :] = feature_matrix[row_idx, :]
    else
        println("⚠️ Missing features for ID: $id_str - filling with zeros")
        missing_count += 1
        # Already zeros from initialization
    end
end

# Create DataFrame
println("Creating aligned DataFrame...")
aligned_df = DataFrame(aligned_matrix, feature_cols)
insertcols!(aligned_df, 1, :image_id => aligned_ids)

# Save the correctly aligned features
CSV.write(joinpath(@__DIR__, "../Data/features_test_CORRECTLY_ALIGNED.csv"), aligned_df)
println("✅ Saved correctly aligned features to: Data/features_test_CORRECTLY_ALIGNED.csv")
println(" Missing features filled: $missing_count")
println(" Total rows: $(nrow(aligned_df))")

# 5. Verify the result
println("\n🔍 Verifying alignment...")
if nrow(aligned_df) == length(correct_ids)
    println("✅ Row count matches test.csv")
else
    println("❌ Row count mismatch!")
end
println("First 5 aligned IDs: $(aligned_df[1:5, "image_id"])")
println("Should match: $(correct_ids[1:5])")

if aligned_df[!, "image_id"] == correct_ids
    println("✅ PERFECT: IDs are now in correct order!")
else
    println("⚠️ IDs may not match exactly")
end

# 6. Do the same for train features
println("\n🔧 Aligning train features...")
if isfile(joinpath(@__DIR__, "../Data/features_train.csv")) && isfile(joinpath(@__DIR__, "../Data/train.csv"))
    println("Loading train data...")
    train_csv = CSV.read(joinpath(@__DIR__, "../Data/train.csv"), DataFrame)
    train_correct_ids = [replace(id, ".png" => "") for id in train_csv.id]
   
    orig_train_features = CSV.read(joinpath(@__DIR__, "../Data/features_train.csv"), DataFrame)
   
    # Convert train features to matrix
    train_feature_cols = [col for col in names(orig_train_features) if col != "image_id"]
    train_feature_matrix = Matrix{Float32}(orig_train_features[!, train_feature_cols])
    train_id_vector = string.(orig_train_features[!, "image_id"])
   
    # Create dictionary for train
    train_id_to_index = Dict{String, Int}()
    for (i, id) in enumerate(train_id_vector)
        train_id_to_index[id] = i
    end
   
    # Build aligned train features
    train_aligned_matrix = zeros(Float32, length(train_correct_ids), length(train_feature_cols))
    train_aligned_ids = Vector{String}()
    train_missing = 0
   
    for (i, id) in enumerate(train_correct_ids)
        id_str = string(id)
        push!(train_aligned_ids, id_str)
       
        if id_str in keys(train_id_to_index)
            row_idx = train_id_to_index[id_str]
            train_aligned_matrix[i, :] = train_feature_matrix[row_idx, :]
        else
            println("⚠️ Missing train features for ID: $id_str")
            train_missing += 1
        end
    end
   
    # Create train DataFrame
    train_aligned_df = DataFrame(train_aligned_matrix, train_feature_cols)
    insertcols!(train_aligned_df, 1, :image_id => train_aligned_ids)
   
    CSV.write(joinpath(@__DIR__, "../Data/features_train_CORRECTLY_ALIGNED.csv"), train_aligned_df)
    println("✅ Saved aligned train features to: Data/features_train_CORRECTLY_ALIGNED.csv")
    println(" Missing train features: $train_missing")
    println(" Total rows: $(nrow(train_aligned_df))")
   
    # Verify train alignment
    if train_aligned_df[!, "image_id"] == train_correct_ids
        println("✅ Train IDs perfectly aligned!")
    else
        println("⚠️ Train IDs may not match exactly")
    end
else
    println("⚠️ Skipping train alignment - files not found")
end

println("\n" * "="^60)
println("🎉 ALIGNMENT FIX COMPLETE!")
println("="^60)
println("""
📋 QUICK NEXT STEPS:
# Update configs and re-run
sed -i 's/Data/features_train_ALIGNED.csv/features_train_CORRECTLY_ALIGNED.csv/' train_final.jl
sed -i 's/Data/features_test_ALIGNED.csv/features_test_CORRECTLY_ALIGNED.csv/' predict.jl
# Re-train and predict
julia train_final.jl
julia predict.jl
""")
