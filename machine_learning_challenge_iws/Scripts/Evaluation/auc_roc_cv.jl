"""
auc_roc_cv.jl
5-fold Stratified Cross-Validation with ROC-AUC scoring
FIXED: Severe overfitting and scoping issues
"""

using DataFrames
using CSV
using XGBoost
using Statistics
using Printf
using Random

# ----------------------------- CONFIG -----------------------------
FEATURES_FILE = "Data/features_train_CORRECTLY_ALIGNED.csv"
TRAIN_METADATA = "Data/train.csv" # Must have columns: id, ground_truth
LABEL_COLUMN = "ground_truth"

# CORRECTED XGBoost parameters to prevent overfitting
XGB_PARAMS = [
    ("tree_method", "hist"),
    ("seed", 1),
    ("learning_rate", 0.01),      # SLOWER learning rate
    ("max_depth", 4),             # SHALLOWER trees
    ("subsample", 0.8),           # Use 80% of data per tree
    ("colsample_bytree", 0.8),    # Use 80% of features per tree
    ("device", "gpu"),
    ("objective", "binary:logistic"),
    ("eval_metric", "auc"),
    # STRONG REGULARIZATION
    ("lambda", 5.0),              # Strong L2 regularization
    ("alpha", 1.0),               # L1 regularization
    ("min_child_weight", 10),     # Require more samples per leaf
    ("gamma", 1.0),               # Minimum loss reduction
    ("max_delta_step", 0)         # Constraint on leaf weights
]

# REDUCE iterations significantly
N_ESTIMATORS = 200
N_FOLDS = 5
CV_RANDOM_STATE = 2024
# ------------------------------------------------------------------

# Manual ROC-AUC implementation (pure Julia, accurate)
function roc_auc_score(y_true::Vector{Float32}, y_pred::Vector{Float32})
    # Sort by predicted score descending
    desc_idx = sortperm(y_pred, rev=true)
    y_true = y_true[desc_idx]
    y_pred = y_pred[desc_idx]
    
    # Compute TPR and FPR at all thresholds
    n_pos = sum(y_true .== 1)
    n_neg = length(y_true) - n_pos
    if n_pos == 0 || n_neg == 0
        return 1.0 # Perfect separation or all one class
    end
    tpr = cumsum(y_true) ./ n_pos
    fpr = cumsum(1 .- y_true) ./ n_neg
    
    # Trapezoidal rule for AUC
    auc = 0.0
    for i in 2:length(fpr)
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    end
    return auc
end

# Simple stratified k-fold indices generator
function stratified_kfold_indices(y::Vector{Float32}, n_folds::Int, random_state::Int)
    Random.seed!(random_state)
    
    # Get indices for each class
    pos_idx = findall(y .== 1)
    neg_idx = findall(y .== 0)
    
    # Shuffle indices within each class
    shuffle!(pos_idx)
    shuffle!(neg_idx)
    
    # Distribute indices across folds
    fold_indices = [Int[] for _ in 1:n_folds]
    
    # Distribute positive samples
    for (i, idx) in enumerate(pos_idx)
        fold = mod(i-1, n_folds) + 1
        push!(fold_indices[fold], idx)
    end
    
    # Distribute negative samples  
    for (i, idx) in enumerate(neg_idx)
        fold = mod(i-1, n_folds) + 1
        push!(fold_indices[fold], idx)
    end
    
    return fold_indices
end

function main()
    println("="^60)
    println("🌊 OCEAN INTERNAL WAVE - 5-FOLD CV (ROC-AUC)")
    println("FIXED: Overfitting prevention + proper regularization")
    println("="^60)

    println("\nLoading features from $FEATURES_FILE...")
    features_df = CSV.read(FEATURES_FILE, DataFrame)
    println("Features: $(nrow(features_df)) samples × $(ncol(features_df)-1) dims")

    # === CRITICAL FIX: Force feature image_id to String ===
    println("\n=== FIXING ID TYPE FOR MERGING ===")
    println("Before: image_id column type in features = ", eltype(features_df.image_id))
    println("Sample feature IDs (before): ", first(features_df.image_id, 5))

    features_df[!, :image_id] = string.(features_df[!, :image_id])  # Convert Int → String

    println("After:  image_id column type in features = ", eltype(features_df.image_id))
    println("Sample feature IDs (after):  ", first(features_df.image_id, 5))
    println("=== FIX APPLIED ===\n")

    println("\nLoading labels from $TRAIN_METADATA...")
    train_meta = CSV.read(TRAIN_METADATA, DataFrame)
    println("Labels: $(nrow(train_meta)) samples")

    # Find and clean the ID column in train.csv
    id_col = nothing
    possible_id_names = ["id", "image_id", "filename"]
    for name in possible_id_names
        if name in names(train_meta)
            id_col = name
            break
        end
    end
    if id_col === nothing
        println("Available columns in train.csv: ", names(train_meta))
        error("No recognizable ID column found. Expected one of: id, image_id, filename")
    end

    # Remove .png extension and ensure String type
    train_meta[!, id_col] = replace.(string.(train_meta[!, id_col]), ".png" => "")
    if id_col != "image_id"
        rename!(train_meta, id_col => "image_id")
    end

    # Extra safety: ensure train image_id is String
    train_meta[!, :image_id] = string.(train_meta[!, :image_id])

    println("Sample train IDs (after cleaning): ", first(train_meta.image_id, 5))

    # Verify label column exists
    if !(LABEL_COLUMN in names(train_meta))
        error("Label column '$LABEL_COLUMN' not found — expected 'ground_truth'")
    end

    println("\nMerging features and labels on image_id...")
    data = innerjoin(features_df, select(train_meta, "image_id", LABEL_COLUMN), on="image_id")
    println("Successfully merged: $(nrow(data)) samples")

    if nrow(data) == 0
        error("Merge failed! Check ID formats above. Expected ~13668 samples.")
    end

    feature_cols = [col for col in names(data) if startswith(string(col), "feature_")]
    X = Matrix{Float32}(data[!, feature_cols])
    y = Float32.(data[!, LABEL_COLUMN])

    println("\nData ready:")
    println(" X: $(size(X))")
    println(" y positives: $(sum(y .== 1)) ($(round(mean(y)*100, digits=2))%)")

    println("\nStarting 5-fold stratified CV...")
    println("⚠️ Using STRONG regularization to prevent overfitting")
    fold_scores = Float64[]

    # Generate stratified folds
    fold_indices = stratified_kfold_indices(y, N_FOLDS, CV_RANDOM_STATE)

    for fold in 1:N_FOLDS
        println("\nFold $fold/$N_FOLDS")
        
        val_idx = fold_indices[fold]
        train_idx = setdiff(1:length(y), val_idx)

        X_train, y_train = X[train_idx, :], y[train_idx]
        X_val, y_val = X[val_idx, :], y[val_idx]

        println("  Train: $(length(y_train)) samples ($(round(mean(y_train)*100, digits=1))% positive)")
        println("  Val: $(length(y_val)) samples ($(round(mean(y_val)*100, digits=1))% positive)")

        dtrain = DMatrix(X_train, label=y_train)
        dval = DMatrix(X_val, label=y_val)

        # Initialize model variable
        model = nothing
        
        try
            println("  Training XGBoost with regularization...")
            model = xgboost(dtrain,
                num_round = N_ESTIMATORS,
                param = XGB_PARAMS,
                verbose_eval = false  # Turn off verbose to see clean output
            )
        catch e
            println("  ⚠️  XGBoost parameter error: ", e)
            println("  Trying alternative parameter format...")
            
            # Alternative: Try Dict format (some XGBoost.jl versions prefer this)
            alt_params = Dict(
                "tree_method" => "hist",
                "seed" => 1,
                "learning_rate" => 0.01,
                "max_depth" => 4,
                "subsample" => 0.8,
                "colsample_bytree" => 0.8,
                "device" => "gpu",
                "objective" => "binary:logistic",
                "eval_metric" => "auc"
            )
            
            model = xgboost(dtrain,
                num_round = 100,
                param = alt_params,
                verbose_eval = false
            )
        end

        if model === nothing
            error("Model training failed for fold $fold")
        end

        val_pred = XGBoost.predict(model, dval)
        if val_pred isa Matrix
            val_pred = vec(val_pred)
        end

        auc = roc_auc_score(y_val, Float32.(val_pred))
        push!(fold_scores, auc)
        println(" Fold $fold AUC: $(@sprintf("%.6f", auc))")
        
        # Show some training diagnostics
        train_pred = XGBoost.predict(model, dtrain)
        if train_pred isa Matrix
            train_pred = vec(train_pred)
        end
        
        train_rmse = sqrt(mean((train_pred .- y_train) .^ 2))
        val_rmse = sqrt(mean((val_pred .- y_val) .^ 2))
        
        println("  Train RMSE: $(@sprintf("%.4f", train_rmse)), Val RMSE: $(@sprintf("%.4f", val_rmse))")
        
        # Check for overfitting
        if train_rmse < 0.01  # If training RMSE is too low
            println("  ⚠️  WARNING: Possible overfitting (train RMSE too low)")
        end
    end

    mean_auc = mean(fold_scores)
    std_auc = std(fold_scores)

    println("\n" * "="^60)
    println("CROSS-VALIDATION RESULTS")
    println("="^60)
    for (i, s) in enumerate(fold_scores)
        println(" Fold $i: $(@sprintf("%.6f", s))")
    end
    println(" Mean AUC: $(@sprintf("%.6f", mean_auc)) ± $(@sprintf("%.6f", std_auc))")
    println("="^60)

    # Performance assessment
    if mean_auc >= 0.95
        println("🎉 Excellent performance! Model is well-regularized.")
    elseif mean_auc >= 0.85
        println("✅ Good performance — model is generalizing well.")
    elseif mean_auc >= 0.70
        println("⚠️ Moderate performance — could be improved.")
    else
        println("❌ Poor performance — check data or model configuration.")
    end
    
    # Overfitting check
    if any(s -> s > 0.98, fold_scores)
        println("⚠️  CAUTION: Some folds have AUC > 0.98 - possible data leakage or overfitting!")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
