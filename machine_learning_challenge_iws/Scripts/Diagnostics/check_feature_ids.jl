# check_feature_ids.jl
using DataFrames, CSV

println("Checking feature file IDs vs actual folder IDs...")

# Get actual folder IDs (from verification we know they're 747454, 393676, etc.)
train_df = CSV.read("Data/train.csv", DataFrame)
actual_train_ids = Set([replace(string(id), ".png" => "") for id in train_df[!, "id"]])

# Check feature file IDs
if isfile("Data/features_train.csv")
    feat_df = CSV.read("Data/features_train.csv", DataFrame)
    if "image_id" in names(feat_df)
        feature_ids = Set(string.(feat_df[!, "image_id"]))
        
        println("\nTRAIN Feature IDs vs Actual IDs:")
        println("   Actual IDs count: $(length(actual_train_ids))")
        println("   Feature IDs count: $(length(feature_ids))")
        println("   Common: $(length(intersect(actual_train_ids, feature_ids)))")
        
        if length(intersect(actual_train_ids, feature_ids)) == 0
            println("\n❌ CRITICAL: NO MATCHING IDs!")
            println("   Actual IDs sample: $(collect(actual_train_ids)[1:5])")
            println("   Feature IDs sample: $(collect(feature_ids)[1:5])")
        else
            match_percent = length(intersect(actual_train_ids, feature_ids)) / length(actual_train_ids) * 100
            println("\n   Match: $(round(match_percent, digits=2))%")
        end
    end
end
