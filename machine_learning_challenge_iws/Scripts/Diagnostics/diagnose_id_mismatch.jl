# clear_diagnostic.jl
using DataFrames, CSV
println("🔍 CLEAR DIAGNOSTIC")
println("="^50)
# 1. Get actual folder contents
# 1. Get actual folder contents (all images are now directly in Data/)
println("\n📂 ACTUAL FOLDER CONTENTS (in Data/):")

data_dir = joinpath(@__DIR__, "../../Data")
all_files = readdir(data_dir)

# Keep only .png image files
png_files = [f for f in all_files if endswith(lowercase(f), ".png")]

println("Total .png images found: $(length(png_files))")
println("First 20 image names: $(sort(png_files)[1:min(20, end)])")
println("Note: All images are directly in Data/ (no train/ or test/ subfolders)")

# 2. Get CSV contents
println("\n📄 CSV FILE CONTENTS:")
train_csv = CSV.read(joinpath(data_dir, "train.csv"), DataFrame)
test_csv = CSV.read(joinpath(data_dir, "test.csv"), DataFrame)
println("First 5 train CSV IDs: $(first(train_csv.id, 5))")
println("First 5 test CSV IDs: $(first(test_csv.id, 5))")

# 3. Get feature file contents
println("\n🎯 FEATURE FILE CONTENTS:")
if isfile(joinpath(data_dir, "features_train.csv"))
    feat_train = CSV.read(joinpath(data_dir, "features_train.csv"), DataFrame)
    println("First column name: $(names(feat_train)[1])")
    println("First 5 IDs: $(first(feat_train[:, 1], 5))")
end

# 4. SIMPLE COMPARISON
println("\n🔍 SIMPLE COMPARISON:")
println("-"^30)
# Convert CSV IDs without .png
csv_train_ids = [replace(id, ".png" => "") for id in train_csv.id]
csv_test_ids = [replace(id, ".png" => "") for id in test_csv.id]

# Convert all image filenames to IDs (remove .png)
folder_all_ids = [replace(f, ".png" => "") for f in png_files]

# OPTION 3: Create separate train/test lists from folder_all_ids
folder_train_ids = filter(id -> id in Set(csv_train_ids), folder_all_ids)
folder_test_ids  = filter(id -> id in Set(csv_test_ids),  folder_all_ids)

println("Comparing first 5 IDs:")
println("CSV Train: $(csv_train_ids[1:5])")
println("Folder Train: $(folder_train_ids[1:min(5, length(folder_train_ids))])")
println("")
println("CSV Test: $(csv_test_ids[1:5])")
println("Folder Test: $(folder_test_ids[1:min(5, length(folder_test_ids))])")

# 5. Check if first ID from CSV exists in folder
println("\n🔎 QUICK EXISTENCE CHECK:")
first_csv_train_id = csv_train_ids[1]
first_csv_test_id = csv_test_ids[1]
println("Does CSV train ID '$first_csv_train_id.png' exist in Data/?")
println(" $(first_csv_train_id * ".png" in png_files ? "✅ YES" : "❌ NO")")

if @isdefined(test_csv)
    println("Does CSV test ID '$first_csv_test_id.png' exist in Data/?")
    println(" $(first_csv_test_id * ".png" in png_files ? "✅ YES" : "❌ NO")")
end

# 6. Show sample of what actually exists
println("\n📊 SAMPLE OF ACTUAL DATA:")
# Safe print even if empty
if length(folder_all_ids) > 0
    println("Sample image IDs from Data/ folder: $(sort(folder_all_ids)[1:min(20, length(folder_all_ids))])")
else
    println("Sample image IDs from Data/ folder: <none found>")
end
