#!/usr/bin/env julia
"""
train_xgboost.jl - Train XGBoost model for ocean internal wave detection
"""

using DataFrames
using CSV
using XGBoost
using Statistics
using Random
using Printf

function main()
    println("="^60)
    println("🌊 OCEAN INTERNAL WAVE DETECTOR - TRAINING")
    println("="^60)
    
    # Load data
    println("\n📊 Loading data...")
    
    println("Loading features_train.csv...")
    features = CSV.read("features_train.csv", DataFrame)
    println("  Samples: $(nrow(features)), Features: $(ncol(features)-1)")
    
    println("Loading labels.csv...")
    labels = CSV.read("labels.csv", DataFrame)
    println("  Samples: $(nrow(labels))")
    
    # Verify order
    println("\nFirst 3 feature IDs:")
    for i in 1:min(3, nrow(features))
        println("  $(features[i, :image_id])")
    end
    
    println("\nFirst 3 label IDs:")
    for i in 1:min(3, nrow(labels))
        println("  $(labels[i, :image_id])")
    end
    
    println("\n⚠️  Assuming features and labels are in same order...")
    
    # Extract feature columns
    feature_cols = [col for col in names(features) if startswith(string(col), "feature_")]
    println("Found $(length(feature_cols)) feature columns")
    
    # Prepare X and y
    X = zeros(Float32, nrow(features), length(feature_cols))
    for (i, col) in enumerate(feature_cols)
        X[:, i] = Float32.(features[!, col])
    end
    
    y = Float32.(labels[!, :label])
    
    println("\n✅ Data prepared!")
    println("  X shape: $(size(X))")
    println("  y shape: $(size(y))")
    println("  Positive samples: $(sum(y .== 1)) ($(round(mean(y)*100, digits=1))%)")
    
    # Split data - CORRECTED SECTION
    println("\n✂️  Splitting data...")
    
    Random.seed!(42)
    n = size(X, 1)
    indices = shuffle(1:n)
    split = Int(floor(0.8 * n))
    
    X_train = X[indices[1:split], :]
    y_train = y[indices[1:split]]
    X_test = X[indices[split+1:end], :]
    y_test = y[indices[split+1:end]]
    
    println("  Training: $(size(X_train, 1)) samples")
    println("  Testing:  $(size(X_test, 1)) samples")
    
    # Train XGBoost
    println("\n🚀 Training XGBoost...")
    
    # XGBoost parameters
    params = Dict(
        "max_depth" => 7,
        "eta" => 0.05,
        "subsample" => 0.50,
        "colsample_bytree" => 0.75,
        "tree_method" => "hist",
        "objective" => "binary:logistic",
        "eval_metric" => "auc",
        "seed" => 1,
        "nthread" => Threads.nthreads()
    )
    
    # Create DMatrix objects
    dtrain = DMatrix(X_train, label=y_train)
    dtest = DMatrix(X_test, label=y_test)
    
    # Train model
    model = xgboost(dtrain, 
        param=params, 
        nrounds=550,
        evals=[(dtest, "eval")],
        verbose_eval=50
    )
    
    # Evaluate
    y_pred = XGBoost.predict(model, dtest)
    accuracy = mean((y_pred .> 0.5) .== y_test)
    
    println("\n✅ Model trained! Accuracy: $(round(accuracy*100, digits=2))%")
    
    # Save model
    println("\n💾 Saving model...")
    XGBoost.save(model, "ocean_wave_model.model")
    println("✅ Model saved: ocean_wave_model.model")
    
    println("\n" * "="^60)
    println("🎉 TRAINING COMPLETE!")
    println("="^60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
