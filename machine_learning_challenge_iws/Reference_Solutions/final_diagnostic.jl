# final_diagnostic.jl
using DataFrames, CSV

println("🎯 FINAL DIAGNOSTIC: Understanding the Mismatch")
println("="^60)

# 1. Get ALL image files in folders
train_files = readdir("Data/train/")
test_files = readdir("Data/test/")

train_png = [f for f in train_files if endswith(f, ".png")]
test_png = [f for f in test_files if endswith(f, ".png")]

println("\n📂 FOLDER STATISTICS:")
println("Train: $(length(train_png)) PNG files")
println("Test: $(length(test_png)) PNG files")

# 2. Get CSV IDs
train_csv = CSV.read("Data/train.csv", DataFrame)
test_csv = CSV.read("Data/test.csv", DataFrame)

csv_train_ids = Set([replace(id, ".png" => "") for id in train_csv.id])
csv_test_ids = Set([replace(id, ".png" => "") for id in test_csv.id])

println("\n📄 CSV STATISTICS:")
println("Train CSV: $(length(csv_train_ids)) IDs")
println("Test CSV: $(length(csv_test_ids)) IDs")

# 3. Get feature file IDs
feat_train = CSV.read("Data/features_train.csv", DataFrame)
feat_test = CSV.read("Data/features_test.csv", DataFrame)

feat_train_ids = Set(string.(feat_train[!, :image_id]))
feat_test_ids = Set(string.(feat_test[!, :image_id]))

println("\n🎯 FEATURE FILE STATISTICS:")
println("Train features: $(length(feat_train_ids)) IDs")
println("Test features: $(length(feat_test_ids)) IDs")

# 4. CRITICAL ANALYSIS
println("\n🔍 CRITICAL ANALYSIS:")
println("-"^40)

# What percentage of CSV IDs are in folders?
folder_train_ids = Set([replace(f, ".png" => "") for f in train_png])
folder_test_ids = Set([replace(f, ".png" => "") for f in test_png])

csv_in_folder_train = intersect(csv_train_ids, folder_train_ids)
csv_in_folder_test = intersect(csv_test_ids, folder_test_ids)

println("CSV IDs found in folders:")
println("  Train: $(length(csv_in_folder_train))/$(length(csv_train_ids)) ($(round(length(csv_in_folder_train)/length(csv_train_ids)*100, digits=1))%)")
println("  Test: $(length(csv_in_folder_test))/$(length(csv_test_ids)) ($(round(length(csv_in_folder_test)/length(csv_test_ids)*100, digits=1))%)")

# What percentage of feature IDs match CSV IDs?
feat_match_csv_train = intersect(feat_train_ids, csv_train_ids)
feat_match_csv_test = intersect(feat_test_ids, csv_test_ids)

println("\nFeature IDs matching CSV IDs:")
println("  Train: $(length(feat_match_csv_train))/$(length(csv_train_ids)) ($(round(length(feat_match_csv_train)/length(csv_train_ids)*100, digits=1))%)")
println("  Test: $(length(feat_match_csv_test))/$(length(csv_test_ids)) ($(round(length(feat_match_csv_test)/length(csv_test_ids)*100, digits=1))%)")

# 5. SAMPLE ANALYSIS
println("\n📊 SAMPLE ANALYSIS (first 20 of each):")
println("First 20 CSV train IDs: $(collect(csv_train_ids)[1:20])")
println("First 20 feature train IDs: $(collect(feat_train_ids)[1:20])")
println("First 20 folder IDs (alphabetical): $(collect(folder_train_ids)[1:20])")

# 6. THE PROBLEM
println("\n" * "="^60)
println("🚨 THE ACTUAL PROBLEM REVEALED:")
println("="^60)

if length(feat_match_csv_train) == 0
    println("""
    ❌ CONFIRMED: Feature extraction ran on WRONG subset of images!
    
    WHAT HAPPENED:
    1. Your folders contain ALL images (100xxx + 600xxx + 700xxx mixed)
    2. Feature extraction processed images in ALPHABETICAL ORDER
    3. It extracted features from the FIRST 13,668 train images (100xxx series)
    4. It extracted features from the FIRST 5,860 test images (100xxx series)
    5. But your CSV files reference DIFFERENT images (random mix)
    
    EXAMPLE:
    - features_train.csv has: 100028, 100047, 100050, etc. (alphabetically first)
    - train.csv has: 603303.png, 618432.png, 851505.png, etc. (random IDs)
    - These are DIFFERENT images!
    
    SOLUTION:
    1. You need to extract features from SPECIFIC images listed in CSV files
    2. NOT from all images in alphabetical order
    """)
end

# 7. How to fix
println("\n🔧 HOW TO FIX YOUR EXTRACT_FEATURES.JL:")
println("-"^40)
println("""
Current approach (WRONG):
  - Reads all images from train/ folder in alphabetical order
  - Extracts features from whatever images it finds first

Required approach (CORRECT):
  - Read train.csv to get list of SPECIFIC image IDs
  - Only extract features from those specific images
  - Same for test.csv

Quick fix for your extract_features.jl:
  1. Load train.csv and test.csv
  2. Use ONLY the images listed in those CSV files
  3. Ignore other images in the folders
""")
