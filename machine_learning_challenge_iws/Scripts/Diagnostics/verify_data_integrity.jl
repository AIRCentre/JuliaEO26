using DataFrames, CSV, Printf, Dates

println("="^80)
println("🔍 HACKATHON DATA DETECTIVE - Scanning project root + Data folder")
println("="^80)
println("Analyzing ALL data files to understand your setup...")
println()

# ============================================================================
# DEFINE DIRECTORIES
# ============================================================================
data_dir = joinpath(@__DIR__, "../../Data")
root_dir = joinpath(@__DIR__, "../..")

# ============================================================================
# INITIALIZE VARIABLES EARLY TO AVOID UndefVarError
# ============================================================================
issues = []
recommendations = []
missing_parts = []

# ============================================================================
# 1. SCAN ALL FILES IN PROJECT ROOT AND Data/ FOLDER
# ============================================================================
println("📁 STEP 1: SCANNING ALL FILES")
println("-"^50)

root_files = readdir(root_dir)
data_files = isdir(data_dir) ? readdir(data_dir) : []
files = unique(vcat(root_files, data_files))

println("Total files: $(length(files))")

categories = Dict(
    "CSV Files" => [],
    "Python Scripts" => [],
    "Julia Scripts" => [],
    "Model Files" => [],
    "Excel Files" => [],
    "Text Files" => [],
    "Other" => []
)

for file in files
    if endswith(file, ".csv")
        push!(categories["CSV Files"], file)
    elseif endswith(file, ".py")
        push!(categories["Python Scripts"], file)
    elseif endswith(file, ".jl")
        push!(categories["Julia Scripts"], file)
    elseif endswith(file, ".model") || endswith(file, ".onnx")
        push!(categories["Model Files"], file)
    elseif endswith(file, ".xlsx") || endswith(file, ".xls")
        push!(categories["Excel Files"], file)
    elseif endswith(file, ".txt") || endswith(file, ".json")
        push!(categories["Text Files"], file)
    else
        push!(categories["Other"], file)
    end
end

for (category, file_list) in categories
    if !isempty(file_list)
        println("📂 $category ($(length(file_list))):")
        for file in sort(file_list)[1:min(5, length(file_list))]
            println("   • $file")
        end
        if length(file_list) > 5
            println("   • ... and $(length(file_list)-5) more")
        end
    end
end
println()

# ============================================================================
# 2. ANALYZE CSV FILES
# ============================================================================
println("📊 STEP 2: ANALYZING CSV FILES")
println("-"^50)

csv_files_in_data = filter(f -> endswith(f, ".csv"), data_files)

if !isempty(csv_files_in_data)
    csv_analysis = []

    for csv_file in sort(csv_files_in_data)
        full_path = joinpath(data_dir, csv_file)
        try
            df = CSV.read(full_path, DataFrame; limit=3)
            n_rows = try
                CSV.read(full_path, DataFrame) |> nrow
            catch
                "Unknown"
            end

            n_cols = size(df, 2)
            col_names = names(df)

            file_type = "Unknown"
            if occursin("feature", lowercase(csv_file))
                file_type = "Feature File"
            elseif occursin("train", lowercase(csv_file))
                file_type = "Training Data"
            elseif occursin("test", lowercase(csv_file))
                file_type = "Test Data"
            elseif occursin("submission", lowercase(csv_file))
                file_type = "Submission File"
            end

            push!(csv_analysis, (csv_file, file_type, n_rows, n_cols, col_names[1:min(3, length(col_names))]))

        catch e
            push!(csv_analysis, (csv_file, "Unreadable ($e)", "Error", "Error", []))
        end
    end

    println("Found $(length(csv_analysis)) CSV files:")
    println()
    println("File Name                      | Type            | Rows    | Cols | Sample Columns")
    println("-"^80)

    for (file, ftype, rows, cols, sample_cols) in csv_analysis
        file_display = length(file) > 30 ? file[1:27] * "..." : file
        cols_display = cols isa Int ? cols : "-"
        rows_display = rows isa Int ? rows : "-"

        sample_str = join(sample_cols[1:min(3, length(sample_cols))], ", ")
        if length(sample_cols) > 3
            sample_str *= ", ..."
        end

        println(@sprintf("%-30s | %-15s | %-7s | %-4s | %s",
            file_display, ftype, rows_display, cols_display, sample_str))
    end
else
    println("No CSV files found!")
end
println()

# ============================================================================
# 3. DETECT COMMON HACKATHON PATTERNS
# ============================================================================
println("🎯 STEP 3: DETECTING COMMON HACKATHON PATTERNS")
println("-"^50)
patterns_found = []

feature_files = filter(x -> occursin("feature", lowercase(x)), csv_files_in_data)
if length(feature_files) > 1
    push!(patterns_found, "⚠️ MULTIPLE FEATURE FILES: $(join(sort(feature_files), ", "))")
end

train_csvs = filter(x -> occursin("train", lowercase(x)) && endswith(x, ".csv"), files)
test_csvs = filter(x -> occursin("test", lowercase(x)) && endswith(x, ".csv"), files)

if !isempty(train_csvs)
    push!(patterns_found, "✅ TRAINING DATA: $(join(sort(train_csvs), ", "))")
else
    push!(patterns_found, "❌ NO TRAINING CSV FOUND")
end

if !isempty(test_csvs)
    push!(patterns_found, "✅ TEST DATA: $(join(sort(test_csvs), ", "))")
else
    push!(patterns_found, "❌ NO TEST CSV FOUND")
end

if !isempty(categories["Model Files"])
    push!(patterns_found, "🤖 MODEL FILES: $(join(sort(categories["Model Files"]), ", "))")
end

submission_files = filter(x -> occursin("submission", lowercase(x)), csv_files_in_data)
if !isempty(submission_files)
    push!(patterns_found, "📤 SUBMISSION FILES: $(join(sort(submission_files), ", "))")
end

for pattern in patterns_found
    println(pattern)
end
println()

# ============================================================================
# 4. CHECK DATA CONSISTENCY
# ============================================================================
println("🔗 STEP 4: CHECKING DATA CONSISTENCY")
println("-"^50)

train_csv = !isempty(train_csvs) ? first(sort(train_csvs)) : nothing
test_csv = !isempty(test_csvs) ? first(sort(test_csvs)) : nothing

if train_csv !== nothing
    try
        train_path = joinpath(data_dir, train_csv)
        train_df = CSV.read(train_path, DataFrame)
        println("📘 $train_csv: $(nrow(train_df)) rows")

        id_cols = filter(x -> occursin("id", lowercase(string(x))), names(train_df))
        if !isempty(id_cols)
            println("   ID column: $(id_cols[1])")
            sample_ids = train_df[1:min(3, nrow(train_df)), Symbol(id_cols[1])]
            println("   Sample IDs: $sample_ids")
            if nrow(train_df) > 0 && endswith(string(sample_ids[1]), ".png")
                println("   📸 IDs have .png extension")
            end
        end

        label_cols = filter(x -> occursin("label", lowercase(string(x))) ||
                             occursin("ground", lowercase(string(x))) ||
                             occursin("target", lowercase(string(x))), names(train_df))
        if !isempty(label_cols)
            println("   Label column: $(label_cols[1])")
            labels = train_df[!, Symbol(label_cols[1])]
            unique_labels = unique(labels)
            println("   Unique labels: $unique_labels")
            if length(unique_labels) == 2
                count_0 = sum(labels .== 0)
                count_1 = sum(labels .== 1)
                println("   Class distribution: $count_0 vs $count_1")
            end
        end
    catch e
        println("❌ Could not read $train_csv: $e")
    end
end

if test_csv !== nothing
    try
        test_path = joinpath(data_dir, test_csv)
        test_df = CSV.read(test_path, DataFrame)
        println("\n📗 $test_csv: $(nrow(test_df)) rows")

        id_cols = filter(x -> occursin("id", lowercase(string(x))), names(test_df))
        if !isempty(id_cols)
            println("   ID column: $(id_cols[1])")
            sample_ids = test_df[1:min(3, nrow(test_df)), Symbol(id_cols[1])]
            println("   Sample IDs: $sample_ids")
        end
    catch e
        println("❌ Could not read $test_csv: $e")
    end
end

if train_csv !== nothing && test_csv !== nothing
    try
        train_rows = nrow(CSV.read(joinpath(data_dir, train_csv), DataFrame))
        test_rows = nrow(CSV.read(joinpath(data_dir, test_csv), DataFrame))
        total = train_rows + test_rows
        train_pct = round(100 * train_rows / total, digits=1)
        test_pct = round(100 * test_rows / total, digits=1)

        println("\n📈 TRAIN/TEST SPLIT:")
        println("   Training: $train_rows rows ($train_pct%)")
        println("   Testing: $test_rows rows ($test_pct%)")
        println("   Total: $total rows")
    catch
        println("\n⚠️ Could not compare train/test sizes")
    end
end
println()

# ============================================================================
# 5. IDENTIFY POTENTIAL ISSUES
# ============================================================================
println("🚨 STEP 5: IDENTIFYING POTENTIAL ISSUES")
println("-"^50)

if length(feature_files) > 3
    push!(issues, "Too many feature files ($(length(feature_files))) - which one to use?")
end

key_files = ["train.csv", "test.csv", "transformer_model.onnx"]
for file in key_files
    if !(file in files)
        push!(issues, "Missing key file: $file")
    end
end

if !isempty(feature_files)
    train_feature_files = filter(x -> occursin("train", lowercase(x)), feature_files)
    if !isempty(train_feature_files)
        try
            feat_file = first(sort(train_feature_files))
            feat_path = joinpath(data_dir, feat_file)
            df = CSV.read(feat_path, DataFrame)

            if "image_id" in names(df)
                println("✅ Feature file '$feat_file' has image_id column")
                println("   Rows: $(nrow(df)), Columns: $(ncol(df))")
                feature_cols = filter(x -> startswith(string(x), "feature_"), names(df))
                if !isempty(feature_cols)
                    println("   Found $(length(feature_cols)) feature columns")
                else
                    push!(issues, "Feature file '$feat_file' has no feature_* columns")
                end
            else
                push!(issues, "Feature file '$feat_file' missing image_id column")
            end
        catch e
            push!(issues, "Cannot read feature file '$feat_file': $e")
        end
    end
end

if !isempty(submission_files)
    try
        sub_file = first(submission_files)
        df = CSV.read(joinpath(data_dir, sub_file), DataFrame)
        println("📤 Submission file '$sub_file': $(nrow(df)) rows")
        if "id" in names(df) && ("predicted" in names(df) || "ground_truth" in names(df) || "label" in names(df))
            println("✅ Has correct columns for submission")
        else
            push!(issues, "Submission file has wrong columns: $(names(df))")
        end
    catch
        push!(issues, "Cannot read submission file")
    end
end

if isempty(issues)
    println("✅ No major issues detected!")
else
    println("Found $(length(issues)) potential issues:")
    for (i, issue) in enumerate(issues)
        println(" $i. $issue")
    end
end
println()

# ============================================================================
# 6. RECOMMENDATIONS
# ============================================================================
println("💡 STEP 6: RECOMMENDATIONS FOR HACKATHON TEAMS")
println("-"^50)

if length(feature_files) > 1
    push!(recommendations, "Clean up duplicate feature files - keep only:")
    push!(recommendations, " • features_train.csv or features_train_CORRECTLY_ALIGNED.csv (for training)")
    push!(recommendations, " • features_test.csv or similar (for prediction)")
end

# Detect pipeline components (including Python)
has_extractor = any(f -> occursin("export_transformer", lowercase(f)) || occursin("extract", lowercase(f)), files)
has_trainer = any(f -> occursin("train", lowercase(f)) && (endswith(f, ".jl") || endswith(f, ".py")), files)
has_predictor = any(f -> occursin("predict", lowercase(f)) || occursin("submission", lowercase(f)), files)

println("ML PIPELINE STATUS:")
println(has_extractor ? " ✅ Feature extraction script" : " ❌ Missing feature extraction script")
println(has_trainer ? " ✅ Training script" : " ❌ Missing training script")
println(has_predictor ? " ✅ Prediction script" : " ❌ Missing prediction script")

if has_extractor && has_trainer && has_predictor
    println("\n🎉 COMPLETE PIPELINE DETECTED!")
else
    if !has_extractor push!(missing_parts, "feature extraction") end
    if !has_trainer push!(missing_parts, "model training") end
    if !has_predictor push!(missing_parts, "prediction") end

    println("\n⚠️ INCOMPLETE PIPELINE")
    println("Missing: $(join(missing_parts, ", "))")
    push!(recommendations, "Develop the missing parts: $(join(missing_parts, ", "))")
end

verification_scripts = filter(x -> occursin("verify", lowercase(x)) || occursin("check", lowercase(x)) || occursin("diag", lowercase(x)), categories["Julia Scripts"])
if !isempty(verification_scripts)
    println("\n🔍 VERIFICATION SCRIPTS FOUND:")
    for script in verification_scripts
        println(" • $script")
    end
    push!(recommendations, "Run verification scripts to check data integrity")
end

if !isempty(recommendations)
    println("\n📋 RECOMMENDATIONS:")
    for (i, rec) in enumerate(recommendations)
        println(" $i. $rec")
    end
end

println()
println("="^80)
println("🔍 ANALYSIS COMPLETE")
println("="^80)

println("\n📊 SUMMARY FOR HACKATHON JUDGES:")
println("• CSV Files: $(length(csv_files_in_data))")
println("• Julia Scripts: $(length(categories["Julia Scripts"]))")
println("• Python Scripts: $(length(categories["Python Scripts"]))")
println("• Model Files: $(length(categories["Model Files"]))")
println("• Issues Found: $(length(issues))")
println("• Pipeline Status: $(isempty(missing_parts) ? "Complete" : "Incomplete")")

if train_csv !== nothing && test_csv !== nothing
    try
        train_size = nrow(CSV.read(joinpath(data_dir, train_csv), DataFrame))
        test_size = nrow(CSV.read(joinpath(data_dir, test_csv), DataFrame))
        println("• Dataset Size: $train_size train, $test_size test")
    catch
        println("• Dataset Size: Could not determine")
    end
end

println("="^80)
