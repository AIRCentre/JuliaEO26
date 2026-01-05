# predict.jl - FINAL SUBMISSION WITH CLIPPING
using DataFrames, CSV, XGBoost, Statistics, Printf

println("="^60)
println("🌊 OCEAN INTERNAL WAVE - FINAL SUBMISSION (CLIPPED)")
println("="^60)

# 1. Load test features
println("\n[1/4] Loading test features...")
test_features = CSV.read("Data/features_test_CORRECTLY_ALIGNED.csv", DataFrame)
println("   Samples: $(nrow(test_features))")

# 2. Load test IDs
println("\n[2/4] Loading test IDs...")
test_meta = CSV.read("Data/test.csv", DataFrame)
println("   Samples: $(nrow(test_meta))")

# 3. Prepare feature matrix
println("\n[3/4] Preparing data...")
feature_cols = [col for col in names(test_features) if col != "image_id"]
X_test = Matrix{Float32}(test_features[!, feature_cols])
println("   Feature matrix: $(size(X_test, 1)) × $(size(X_test, 2))")

# 4. Load the trained model
println("\n[4/4] Loading model...")
model_file = "Data/final_xgboost_correctly_aligned.model"

if !isfile(model_file)
    # Try alternative model names
    model_files = filter(x -> occursin(r"\.model$", x), readdir())
    if isempty(model_files)
        error("❌ No model files found!")
    end
    model_file = model_files[1]
    println("   Using alternative model: $model_file")
end

# Load the model
model = XGBoost.load(Booster, model_file)
println("   ✅ Model loaded: $model_file")

# 5. Make predictions
println("\n🎯 Making predictions...")
predictions = XGBoost.predict(model, X_test)
println("   ✅ Raw predictions generated: $(length(predictions))")

# 6. CLIP PREDICTIONS to [0, 1] range
println("\n⚡ CLIPPING predictions to [0, 1] range...")
before_min = minimum(predictions)
before_max = maximum(predictions)

# Apply clipping
predictions_clipped = clamp.(predictions, 0.0, 1.0)

clipped_count = sum((predictions .< 0) .| (predictions .> 1))
if clipped_count > 0
    println("   ⚠️  Clipped $clipped_count values:")
    println("     Before: min=$(@sprintf("%.6f", before_min)), max=$(@sprintf("%.6f", before_max))")
    println("     After:  min=$(@sprintf("%.6f", minimum(predictions_clipped))), max=$(@sprintf("%.6f", maximum(predictions_clipped)))")
else
    println("   ✅ No clipping needed - all values already in [0, 1]")
end

# 7. Show statistics
println("\n📊 Final prediction statistics:")
println("   Min:  $(@sprintf("%.6f", minimum(predictions_clipped)))")
println("   Max:  $(@sprintf("%.6f", maximum(predictions_clipped)))")
println("   Mean: $(@sprintf("%.6f", mean(predictions_clipped)))")
println("   Std:  $(@sprintf("%.6f", std(predictions_clipped)))")

# 8. Create submission file
println("\n💾 Creating submission file...")
submission = DataFrame(
    id = test_meta[!, "id"],  # Keep original IDs with .png
    ground_truth = predictions_clipped
)

output_file = "Data/final_submission.csv"
CSV.write(output_file, submission)
println("   ✅ Submission saved: $output_file")

# 9. Show sample
println("\n📝 Sample of submission (first 5 rows):")
for i in 1:min(5, nrow(submission))
    println("   $(submission[i, "id"]), $(@sprintf("%.6f", submission[i, "ground_truth"]))")
end

# 10. Quality check
println("\n🔍 Quality check:")
# Check if values are sensible
if any(predictions_clipped .< 0) || any(predictions_clipped .> 1)
    println("   ❌ ERROR: Values still outside [0, 1]!")
else
    println("   ✅ All predictions in valid range [0, 1]")
end

# Check distribution
pos_ratio = mean(predictions_clipped .> 0.5)
println("   Predicted positive ratio (>0.5): $(@sprintf("%.1f%%", pos_ratio*100))")

# 11. Final verification
println("\n" * "="^60)
println("✅ FINAL VERIFICATION")
println("="^60)

if isfile(output_file)
    final_check = CSV.read(output_file, DataFrame)
    println("File: $output_file")
    println("Rows: $(nrow(final_check))")
    println("Columns: $(names(final_check))")
    
    if names(final_check) == ["id", "ground_truth"]
        println("✅ Format correct")
    else
        println("⚠️  Format may need adjustment")
    end
    
    # Check for any issues
    if any(isnan.(final_check[!, "ground_truth"]))
        println("⚠️  WARNING: NaN values found in predictions")
    else
        println("✅ No NaN values in predictions")
    end
    
    # Check range
    if all(0 .<= final_check[!, "ground_truth"] .<= 1)
        println("✅ All predictions in [0, 1] range")
    else
        bad_min = minimum(final_check[!, "ground_truth"])
        bad_max = maximum(final_check[!, "ground_truth"])
        println("❌ ERROR: Predictions outside [0, 1]: min=$bad_min, max=$bad_max")
    end
else
    println("❌ ERROR: Submission file not created!")
end

println("\n" * "="^60)
println("🎉 FINAL SUBMISSION READY!")
println("="^60)
println("\nFile: $output_file")
println("Rows: $(nrow(submission))")
println("Positive ratio: $(@sprintf("%.1f%%", pos_ratio*100))")
println("="^60)
