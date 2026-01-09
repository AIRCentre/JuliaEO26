using PyCall
using Images
using DataFrames
using CSV
using Printf
using Dates
using ProgressMeter
using ImageTransformations: imresize

# -------------------------------
# Helper Functions
# -------------------------------

function print_menu()
    println("\n" * "="^60)
    println("🌊 OCEAN INTERNAL WAVE DETECTION - FEATURE EXTRACTION")
    println("="^60)
    println("\nSelect extraction mode:")
    println(" 1. Extract from TRAIN images only")
    println(" 2. Extract from TEST images only")
    println(" 3. Extract from BOTH train and test")
    println(" 4. Exit")
    print("\nEnter choice (1-4): ")
end

function get_user_choice()
    while true
        try
            choice = parse(Int, readline())
            if 1 ≤ choice ≤ 4
                return choice
            else
                println("Please enter 1, 2, 3, or 4")
            end
        catch
            println("Invalid input. Please enter a number (1-4)")
        end
    end
end

function get_max_images()
    print("\nEnter number of images to process (0 for all, default=0): ")
    try
        input = readline()
        if isempty(input)
            return 0
        end
        n = parse(Int, input)
        return n > 0 ? n : 0
    catch
        println("Using default: all images")
        return 0
    end
end

function check_hardware()
    println("\n🔍 Checking hardware...")
    try
        gpu_info = read(`nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`, String)
        println("✅ GPU detected: $gpu_info")
        return true
    catch
        println("⚠️ GPU not detected - using CPU")
        return false
    end
end

# -------------------------------
# ONNX Session Creation
# -------------------------------

function create_session(model_path)
    ort = pyimport("onnxruntime")
   
    println("\nInitializing ONNX Runtime...")
    providers = ort.get_available_providers()
    println("Available providers: $providers")
   
    if "CUDAExecutionProvider" in providers
        println("🎯 Using GPU acceleration")
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = false
        sess_options.enable_mem_pattern = false
        session = ort.InferenceSession(model_path,
                                      sess_options,
                                      providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        device = "GPU"
    else
        println("⚙️ Using CPU")
        session = ort.InferenceSession(model_path,
                                      providers=["CPUExecutionProvider"])
        device = "CPU"
    end
   
    input_name = session.get_inputs()[1].name
    output_name = session.get_outputs()[1].name
   
    println("✅ Session created")
    println(" Device: $device")
    println(" Input: $input_name")
    println(" Output: $output_name")
   
    return session, input_name, output_name, device
end

# -------------------------------
# Image Preprocessing
# -------------------------------

function preprocess_image(img_path::String)
    try
        img = load(img_path)
        
        # Convert to RGB if necessary
        if size(img) == (256, 256) && eltype(img) <: Gray
            img_array = Float32.(channelview(img))
            img_array = repeat(img_array, outer=(1,1,3))
        else
            img_array_raw = channelview(img)
            if ndims(img_array_raw) == 2
                img_array = Float32.(img_array_raw)
                img_array = repeat(img_array, outer=(1,1,3))
            elseif size(img_array_raw,1) == 4
                img_array = Float32.(img_array_raw[1:3,:,:])
            else
                img_array = Float32.(img_array_raw)
            end
        end
        
        img_array = permutedims(img_array, (3,1,2))
        
        # Resize to 448x448
        resized_channels = []
        for c in 1:3
            push!(resized_channels, imresize(img_array[c,:,:], (448,448)))
        end
        resized_array = cat(resized_channels..., dims=3)
        resized_array = permutedims(resized_array, (3,1,2))
        return reshape(resized_array, 1,3,448,448)
        
    catch e
        println("⚠️ Failed to process $img_path: $(typeof(e)) - $(e)")
        return nothing
    end
end

# -------------------------------
# Feature Extraction
# -------------------------------

function extract_from_directory(dir_path::String, session_info, max_images::Int)
    session, input_name, output_name, device = session_info
    println("\n📁 Processing directory: $dir_path")
   
    if !isdir(dir_path)
        println("❌ Directory not found: $dir_path")
        return DataFrame(), 0
    end
   
    image_files = filter(x -> endswith(lowercase(x), ".png"), readdir(dir_path, join=true))
    sort!(image_files)
   
    if max_images > 0 && max_images < length(image_files)
        image_files = image_files[1:max_images]
        println("Limiting to $max_images images")
    end
   
    total_images = length(image_files)
    println("Found $total_images images")
    if total_images == 0
        return DataFrame(), 0
    end
   
    all_features = Vector{Vector{Float32}}()
    all_ids = Vector{String}()
    failed_count = 0
    batch_size = device == "GPU" ? 4 : 16
    total_batches = ceil(Int, total_images / batch_size)
    p = Progress(total_batches, 1, "Extracting...")
    start_time = time()
   
    for i in 1:batch_size:total_images
        batch_end = min(i + batch_size - 1, total_images)
        batch_files = image_files[i:batch_end]
        batch_tensors = Vector{Array{Float32,4}}()
        batch_ids = Vector{String}()
       
        for file in batch_files
            tensor = preprocess_image(file)
            if !isnothing(tensor)
                push!(batch_tensors, tensor)
                push!(batch_ids, replace(basename(file), ".png" => ""))  # FIXED: push to batch_ids
            else
                failed_count += 1
            end
        end
       
        if !isempty(batch_tensors)
            try
                np = pyimport("numpy")
                batch_array = cat(batch_tensors..., dims=1)
                input_data = np.array(batch_array, dtype=np.float32)
                outputs = session.run([output_name], Dict(input_name => input_data))
                batch_features = outputs[1]
               
                # Process successful features
                for j in 1:min(length(batch_ids), size(batch_features,1))
                    push!(all_features, vec(batch_features[j,:]))
                    push!(all_ids, batch_ids[j])  # FIXED: use batch_ids here
                end
            catch e
                # Handle memory errors
                if occursin("CUDA out of memory", string(e)) || occursin("Failed to allocate", string(e))
                    println("\n⚠️ Memory error. Reducing batch size...")
                    # Process in smaller chunks
                    for k in 1:length(batch_tensors)
                        try
                            single_array = reshape(batch_tensors[k], 1, 3, 448, 448)
                            np = pyimport("numpy")
                            input_data = np.array(single_array, dtype=np.float32)
                            outputs = session.run([output_name], Dict(input_name => input_data))
                            single_features = outputs[1]
                            push!(all_features, vec(single_features[1,:]))
                            push!(all_ids, batch_ids[k])
                        catch e2
                            println("⚠️ Failed individual image: $(batch_ids[k])")
                            failed_count += 1
                        end
                    end
                else
                    rethrow(e)
                end
            end
        end
       
        ProgressMeter.update!(p, ceil(Int, batch_end / batch_size))
        if i % (batch_size*10) == 0
            GC.gc()
        end
    end
   
    if !isempty(all_features)
        # Create DataFrame efficiently
        df = DataFrame()
        df[!, "image_id"] = all_ids
        
        feature_dim = length(all_features[1])
        println("Feature dimension: $feature_dim")
        println("Number of images processed: $(length(all_features))")
        
        # Add feature columns
        for i in 1:feature_dim
            df[!, "feature_$i"] = [features[i] for features in all_features]
        end
        
        elapsed = time() - start_time
        speed = length(all_features) / elapsed
       
        println("\n✅ Extraction complete: $dir_path")
        println(" Processed: $(length(all_features)) images")
        println(" Failed: $failed_count images")
        println(" Time: $(round(elapsed,digits=1)) seconds")
        println(" Speed: $(round(speed,digits=1)) img/s")
        println(" Device: $device")
        return df, length(all_features)
    else
        println("❌ No features extracted from $dir_path")
        return DataFrame(), 0
    end
end

# -------------------------------
# Save Features (with train.csv alignment)
# -------------------------------

function save_features(df::DataFrame, filename::String; align_with_train_csv::Bool=false, train_csv_path="")
    if nrow(df) > 0
        if align_with_train_csv && train_csv_path != "" && isfile(train_csv_path)
            println("🔧 Auto-aligning features with train.csv order...")
            try
                train_labels = CSV.read(train_csv_path, DataFrame)
                train_labels[!, :clean_id] = replace.(train_labels[!, :id], ".png"=>"")
                df[!, :image_id] = string.(df[!, :image_id])
                train_labels[!, :clean_id] = string.(train_labels[!, :clean_id])
                
                # Check for duplicates
                if length(unique(df[!, :image_id])) != length(df[!, :image_id])
                    println("⚠️ Duplicate IDs found in features")
                end
                
                ordered_features = DataFrame()
                matched_count = 0
                missing_ids = []
                
                for train_id in train_labels[!, :clean_id]
                    idx = findfirst(df[!, :image_id] .== train_id)
                    if !isnothing(idx)
                        push!(ordered_features, df[idx, :])
                        matched_count += 1
                    else
                        push!(missing_ids, train_id)
                    end
                end
                
                if matched_count > 0
                    df = ordered_features
                    println("✅ $matched_count/$nrow(train_labels) features aligned with train.csv")
                    if !isempty(missing_ids) && length(missing_ids) <= 5
                        println("   Missing IDs in features: $(missing_ids)")
                    end
                else
                    println("⚠️ No matches found - using original order")
                end
            catch e
                println("⚠️ Alignment failed: $e - using original order")
            end
        end
        CSV.write(filename, df)
        println("💾 Saved: $filename ($(round(filesize(filename)/1024^2,digits=2)) MB)")
    end
end

# -------------------------------
# Main
# -------------------------------

function main()
    println("Starting Ocean Internal Wave Feature Extraction...")
    
    # Point to Data folder
    project_data_dir = joinpath(@__DIR__,"../../Data")
    model_path = joinpath(project_data_dir,"transformer_model.onnx")
    if !isfile(model_path)
        println("❌ Error: transformer_model.onnx not found at $model_path")
        return
    end
    
    has_gpu = check_hardware()
    session_info = create_session(model_path)
    
    while true
        print_menu()
        choice = get_user_choice()
        if choice == 4
            println("\n👋 Exiting...")
            break
        end
        max_images = get_max_images()
        train_dir = joinpath(project_data_dir,"train")
        test_dir  = joinpath(project_data_dir,"test")
        train_csv_path = joinpath(project_data_dir,"train.csv")
        
        if choice == 1
            df, count = extract_from_directory(train_dir, session_info, max_images)
            if count > 0
                save_features(df, joinpath(project_data_dir,"features_train.csv"), 
                            align_with_train_csv=true, train_csv_path=train_csv_path)
            else
                println("⚠️ No features extracted from train directory")
            end
        elseif choice == 2
            df, count = extract_from_directory(test_dir, session_info, max_images)
            if count > 0
                save_features(df, joinpath(project_data_dir,"features_test.csv"))
            else
                println("⚠️ No features extracted from test directory")
            end
        elseif choice == 3
            println("\n📊 Extracting from TRAIN images...")
            df_train, count_train = extract_from_directory(train_dir, session_info, max_images)
            if count_train > 0
                save_features(df_train, joinpath(project_data_dir,"features_train.csv"), 
                            align_with_train_csv=true, train_csv_path=train_csv_path)
            else
                println("⚠️ No features extracted from train directory")
            end
            
            println("\n📊 Extracting from TEST images...")
            df_test, count_test = extract_from_directory(test_dir, session_info, max_images)
            if count_test > 0
                save_features(df_test, joinpath(project_data_dir,"features_test.csv"))
            else
                println("⚠️ No features extracted from test directory")
            end
            
            if count_train > 0 && count_test > 0
                println("\n🔗 Combining train and test features...")
                df_all = vcat(df_train, df_test)
                save_features(df_all, joinpath(project_data_dir,"features_all.csv"))
                println("✅ Combined features saved: features_all.csv")
            end
            
            println("\n📈 SUMMARY:")
            println(" Train images: $count_train")
            println(" Test images: $count_test")
            println(" Total images: $(count_train+count_test)")
        end
        println("\n" * "="^60)
        println("🎉 Extraction complete!")
        println("="^60)
        println("\nPress Enter to continue or 'q' to quit...")
        input = readline()
        if lowercase(input) == "q"
            break
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
