using CSV, DataFrames

println("=== ULTIMATE DATA DISCOVERY ===")

train = CSV.read(joinpath(@__DIR__, "../../Data/train.csv"), DataFrame)
features = CSV.read(joinpath(@__DIR__, "../../Data/features_train.csv"), DataFrame)

# Clean train IDs
train_ids = [parse(Int, replace(id, ".png" => "")) for id in train.id]

println("1. Checking if ALL train IDs are in features_train.csv...")
all_found = true
missing_ids = []
for train_id in train_ids[1:20]  # Check first 20
    if !(train_id in features.image_id)
        all_found = false
        push!(missing_ids, train_id)
    end
end

if all_found
    println("   ✅ ALL first 20 train IDs found in features_train.csv")
else
    println("   ❌ Missing IDs: $missing_ids")
end

println("\n2. Checking the ORDER mismatch:")
println("   Train ID 603303 is at row 1 in train.csv")
idx_in_features = findfirst(x -> x == 603303, features.image_id)
println("   Train ID 603303 is at row $idx_in_features in features_train.csv")

println("\n3. Are they just shuffled (same set, different order)?")
set_train = Set(train_ids)
set_features = Set(features.image_id)
println("   Same set? $(set_train == set_features)")
println("   Size of train set: $(length(set_train))")
println("   Size of features set: $(length(set_features))")
println("   Intersection size: $(length(intersect(set_train, set_features)))")

println("\n4. Let's find a few more matches:")
for i in 1:5
    train_id = train_ids[i]
    idx = findfirst(x -> x == train_id, features.image_id)
    println("   Train ID $train_id (train row $i) -> features row $idx")
end

println("\n🚨 CONCLUSION:")
if set_train == set_features
    println("   ✅ features_train.csv HAS ALL train IDs, just SHUFFLED!")
    println("   Your alignment script created features_train_CORRECTLY_ALIGNED.csv")
    println("   which puts them in the CORRECT ORDER!")
else
    println("   ❌ Something else is wrong - sets don't match")
end

