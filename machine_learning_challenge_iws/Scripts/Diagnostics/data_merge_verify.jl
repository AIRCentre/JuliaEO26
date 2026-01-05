# Run this FIRST to see if data can merge
using CSV, DataFrames

println("=== CHECK 1: CAN DATA MERGE? ===")
features = CSV.read("Data/features_train_CORRECTLY_ALIGNED.csv", DataFrame)
train = CSV.read("Data/train.csv", DataFrame)

println("1. Original data:")
println("   Features ID type: $(eltype(features.image_id)), sample: $(features.image_id[1])")
println("   Train ID type: $(eltype(train.id)), sample: $(train.id[1])")

println("\n2. Clean IDs like your script does:")
features.image_id = string.(features.image_id)
train.image_id = replace.(string.(train.id), ".png" => "")

println("   After cleaning:")
println("   Features: $(features.image_id[1]) (type: $(eltype(features.image_id)))")
println("   Train: $(train.image_id[1]) (type: $(eltype(train.image_id)))")

println("\n3. Try merge:")
combined = innerjoin(features, select(train, "image_id", "ground_truth"), on="image_id")
println("   Merge result: $(nrow(combined)) rows")
if nrow(combined) > 0
    println("   ✅ SUCCESS! First row ID: $(combined.image_id[1]), label: $(combined.ground_truth[1])")
else
    println("   ❌ FAILED - 0 rows merged")
    
    println("\n4. Debug why:")
    println("   Checking first 5 IDs match?")
    for i in 1:5
        println("   Row $i: features=$(features.image_id[i]), train=$(train.image_id[i]), match? $(features.image_id[i] == train.image_id[i])")
    end
end

