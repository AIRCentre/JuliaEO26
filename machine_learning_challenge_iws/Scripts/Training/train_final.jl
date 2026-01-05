"""
train_final.jl - CORRECTED VERSION WITH ALIGNED DATA
Train the final XGBoost model on ALL training data using ALIGNED features and labels
"""

using DataFrames
using CSV
using XGBoost
using Printf
using Statistics

# ----------------------------- CONFIG -----------------------------
FEATURES_FILE = "Data/features_train_CORRECTLY_ALIGNED.csv"  # Use ALIGNED features
TRAIN_LABELS_FILE = "Data/train_labels_ALIGNED.csv" # Use ALIGNED labels
LABEL_COLUMN = "ground_truth"

XGB_PARAMS = Dict(
    :tree_method       => "hist",
    :random_state      => 1,
    :learning_rate     => 0.05,
    :max_depth         => 7,
    :subsample         => 0.50,
    :colsample_bytree  => 0.75,
    :device            => "gpu",
    :objective         => "binary:logistic"
)

N_ESTIMATORS = 550
MODEL_OUTPUT = "final_xgboost_correctly_aligned.model"
# ------------------------------------------------------------------

function main()
    println("="^60)
    println("🌊 OCEAN INTERNAL WAVE - FINAL MODEL TRAINING (ALIGNED)")
    println("="^60)

    # Load ALIGNED features
    println("\nLoading ALIGNED features from $FEATURES_FILE...")
    if !isfile(FEATURES_FILE)
        println("❌ ERROR: $FEATURES_FILE not found!")
        println("\nPlease run fix_data_alignment.jl first:")
        println("   julia fix_data_alignment.jl")
        return
    end
    
    features_df = CSV.read(FEATURES_FILE, DataFrame)
    println("✅ Features: $(nrow(features_df)) samples × $(ncol(features_df)-1) dims")
    println("   First feature ID: $(features_df[1, "image_id"])")
    println("   Last feature ID: $(features_df[end, "image_id"])")

    # Load ALIGNED labels
    println("\nLoading ALIGNED labels from $TRAIN_LABELS_FILE...")
    if !isfile(TRAIN_LABELS_FILE)
        println("❌ ERROR: $TRAIN_LABELS_FILE not found!")
        println("\nPlease run fix_data_alignment.jl first:")
        println("   julia fix_data_alignment.jl")
        return
    end
    
    train_labels = CSV.read(TRAIN_LABELS_FILE, DataFrame)
    println("✅ Labels: $(nrow(train_labels)) samples")
    println("   First label ID: $(train_labels[1, "image_id"])")
    println("   Last label ID: $(train_labels[end, "image_id"])")

    # Quick verification
    println("\n🔍 Verifying alignment...")
    if features_df[!, "image_id"] == train_labels[!, "image_id"]
        println("✅ PERFECT: Features and labels are perfectly aligned!")
    else
        println("⚠️  WARNING: IDs don't match exactly")
        println("   Checking first 5 IDs:")
        for i in 1:min(5, nrow(features_df))
            feat_id = features_df[i, "image_id"]
            label_id = train_labels[i, "image_id"]
            match = feat_id == label_id ? "✅" : "❌"
            println("   Row $i: Features=$feat_id, Labels=$label_id $match")
        end
    end

    # Extract X and y directly (since they should be aligned)
    println("\nPreparing data for training...")
    feature_cols = [col for col in names(features_df) if col != "image_id"]
    X = Matrix{Float32}(features_df[!, feature_cols])
    y = Float32.(train_labels[!, "ground_truth"])

    num_samples = size(X, 1)
    num_features = size(X, 2)
    
    println("   Feature matrix X: $(num_samples) × $(num_features)")
    println("   Labels y: $(length(y))")
    println("   Positive samples: $(sum(y .== 1)) ($(round(mean(y)*100, digits=2))%)")
    println("   Negative samples: $(sum(y .== 0))")

    # Check for any NaN/Inf values
    println("\n🔍 Data quality check...")
    if any(isnan.(X))
        println("⚠️  WARNING: NaN values found in features!")
        println("   Replacing with column means...")
        for col in 1:size(X, 2)
            col_data = X[:, col]
            if any(isnan.(col_data))
                col_mean = mean(col_data[.!isnan.(col_data)])
                X[isnan.(col_data), col] = col_mean
            end
        end
    end
    
    if any(isnan.(y))
        println("❌ ERROR: NaN values found in labels!")
        return
    end

    # Create DMatrix
    dtrain = DMatrix(X, label=y)

    # Train final model on ALL data
    println("\n" * "="^60)
    println("🚀 TRAINING FINAL XGBOOST MODEL (ALIGNED)")
    println("   Device: GPU")
    println("   Rounds: $N_ESTIMATORS")
    println("   Samples: $(num_samples)")
    println("   Features: $(num_features)")
    println("="^60)
    
    model = xgboost(dtrain;
        param = XGB_PARAMS,
        num_round = N_ESTIMATORS,
        verbose_eval = 50
    )

    # Save model
    println("\n💾 Saving final model...")
    XGBoost.save(model, MODEL_OUTPUT)
    println("✅ Final model saved as: $MODEL_OUTPUT")
    
    # Quick validation (in-sample predictions)
    println("\n📊 Model validation (in-sample)...")
    predictions = XGBoost.predict(model, dtrain)
    
    # Convert probabilities to binary predictions
    threshold = 0.5
    binary_preds = predictions .>= threshold
    
    # Calculate metrics
    accuracy = mean(binary_preds .== y)
    precision = sum((binary_preds .== 1) .& (y .== 1)) / max(1, sum(binary_preds .== 1))
    recall = sum((binary_preds .== 1) .& (y .== 1)) / max(1, sum(y .== 1))
    f1 = 2 * (precision * recall) / (precision + recall)
    
    println("   Accuracy: $(round(accuracy*100, digits=2))%")
    println("   Precision: $(round(precision*100, digits=2))%")
    println("   Recall: $(round(recall*100, digits=2))%")
    println("   F1 Score: $(round(f1*100, digits=2))%")
    
    # Show prediction distribution
    println("\n📈 Prediction distribution:")
    println("   Min probability: $(minimum(predictions))")
    println("   Max probability: $(maximum(predictions))")
    println("   Mean probability: $(mean(predictions))")
    println("   Std probability: $(std(predictions))")

    println("\n" * "="^60)
    println("🎉 FINAL MODEL TRAINING COMPLETE!")
    println("="^60)
    println("\n📋 NEXT STEPS:")
    println("1. Create predictions using aligned test features:")
    println("   julia predict_aligned.jl")
    println("\n2. The aligned test features are in: features_test_ALIGNED.csv")
    println("3. Test IDs are in: test.csv")
    println("="^60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
