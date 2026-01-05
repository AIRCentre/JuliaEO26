# fix_alignment.jl
using DataFrames, CSV

println("🔧 FIXING ID ALIGNMENT")
println("="^50)

# 1. Load all data
println("\n📊 Loading data...")
train_labels = CSV.read("Data/train.csv", DataFrame)
test_labels = CSV.read("Data/test.csv", DataFrame)
feat_train = CSV.read("Data/features_train.csv", DataFrame)
feat_test = CSV.read("Data/features_test.csv", DataFrame)

# 2. Prepare IDs (remove .png extension)
println("\n📝 Preparing IDs...")
train_label_ids = [replace(id, ".png" => "") for id in train_labels.id]
test_label_ids = [replace(id, ".png" => "") for id in test_labels.id]

train_feat_ids = string.(feat_train.image_id)
test_feat_ids = string.(feat_test.image_id)

# 3. Check current alignment
println("\n🔍 Current alignment check:")
println("First 5 label IDs: $(train_label_ids[1:5])")
println("First 5 feature IDs: $(train_feat_ids[1:5])")

# 4. Create mapping from feature order to label order
println("\n🎯 Creating alignment mapping...")

# Create DataFrames for merging
train_feat_df = DataFrame(image_id=train_feat_ids)
for col in names(feat_train)[2:end]
    train_feat_df[!, col] = feat_train[!, col]
end

test_feat_df = DataFrame(image_id=test_feat_ids)
for col in names(feat_test)[2:end]
    test_feat_df[!, col] = feat_test[!, col]
end

# Merge features with labels to align them
println("Merging train features with labels...")
train_aligned = innerjoin(
    DataFrame(image_id=train_label_ids, ground_truth=train_labels.ground_truth),
    train_feat_df,
    on=:image_id
)

println("Merging test features with labels...")
test_aligned = innerjoin(
    DataFrame(image_id=test_label_ids),
    test_feat_df,
    on=:image_id
)

# 5. Save aligned features
println("\n💾 Saving aligned features...")

# Save aligned train features (without ground_truth for features file)
train_aligned_features = select(train_aligned, Not(:ground_truth))
CSV.write("Data/features_train_ALIGNED.csv", train_aligned_features)
println("Saved features_train_ALIGNED.csv with $(nrow(train_aligned_features)) rows")

# Save aligned test features
CSV.write("Data/features_test_ALIGNED.csv", test_aligned)
println("Saved features_test_ALIGNED.csv with $(nrow(test_aligned)) rows")

# Save aligned labels (just for reference)
CSV.write("Data/train_labels_ALIGNED.csv", select(train_aligned, [:image_id, :ground_truth]))
println("Saved train_labels_ALIGNED.csv with $(nrow(train_aligned)) rows")

# 6. Verify alignment
println("\n✅ VERIFICATION:")
println("First 5 aligned train IDs: $(first(train_aligned.image_id, 5))")
println("Should match: $(first(train_label_ids, 5))")

if first(train_aligned.image_id, 5) == first(train_label_ids, 5)
    println("🎉 SUCCESS: Alignment fixed!")
else
    println("⚠️  Warning: Alignment may still need adjustment")
end

# 7. Create a simple test
println("\n🔬 Simple test:")
println("Checking if IDs are now in the same order...")

all_match_train = all(train_aligned.image_id .== train_label_ids)
all_match_test = all(test_aligned.image_id .== test_label_ids)

println("Train IDs match order? $all_match_train")
println("Test IDs match order? $all_match_test")

if all_match_train && all_match_test
    println("\n🎊 PERFECT! All IDs are now properly aligned!")
    println("\nNext steps:")
    println("1. Use features_train_ALIGNED.csv and features_test_ALIGNED.csv")
    println("2. Train your model with these aligned features")
    println("3. Make sure to keep the same order for predictions")
else
    println("\n⚠️  Some IDs may be missing or out of order")
    println("Check the merged files for any issues")
end
