using PyCall
using Images
using DataFrames
using CSV
using Printf
using Dates
using ProgressMeter
using ImageTransformations: imresize

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

function create_session(model_path="Data/transformer_model.onnx")
    ort = pyimport("onnxruntime")
   
    println("\nInitializing ONNX Runtime...")
    providers = ort.get_available_providers()
    println("Available providers: $providers")
   
    if "CUDAExecutionProvider" in providers
        println("🎯 Using GPU acceleration")
        
        # Set session options to limit GPU memory usage
        sess_options = ort.SessionOptions()
        # Allow memory growth instead of pre-allocating everything
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

function preprocess_image(img_path::String)
    try
        # Load image
        img = load(img_path)
        
        # Convert to RGB if necessary
        if size(img) == (256, 256) && eltype(img) <: Gray
            # Handle grayscale images
            img_array = Float32.(channelview(img))
            # Convert to 3 channels
            img_array = repeat(img_array, outer=(1, 1, 3))
        else
            # Convert to array and ensure 3 channels
            img_array_raw = channelview(img)
            
            # Convert to Float32
            if ndims(img_array_raw) == 2  # Grayscale
                img_array = Float32.(img_array_raw)
                img_array = repeat(img_array, outer=(1, 1, 3))
            elseif size(img_array_raw, 1) == 4  # RGBA
                img_array = Float32.(img_array_raw[1:3, :, :])
            else  # RGB or other
                img_array = Float32.(img_array_raw)
            end
        end
        
        # Permute to (channels, height, width)
        img_array = permutedims(img_array, (3, 1, 2))
        
        # Get current dimensions
        _, h, w = size(img_array)
        
        # Resize to 448x448 using imresize
        # We need to resize each channel separately
        resized_channels = []
        for c in 1:3
            channel_img = img_array[c, :, :]
            resized_channel = imresize(channel_img, (448, 448))
            push!(resized_channels, resized_channel)
        end
        
        # Stack channels back together
        resized_array = cat(resized_channels..., dims=3)
        resized_array = permutedims(resized_array, (3, 1, 2))
        
        # Add batch dimension and return
        return reshape(resized_array, 1, 3, 448, 448)
        
    catch e
        println("⚠️ Failed to process $img_path: $(typeof(e)) - $(e)")
        return nothing
    end
end

function extract_from_directory(dir_name::String, session_info, max_images::Int)
    session, input_name, output_name, device = session_info
   
    println("\n📁 Processing directory: $dir_name")
   
    if !isdir(dir_name)
        println("❌ Directory not found: $dir_name")
        return DataFrame(), 0
    end
   
    image_files = filter(x -> endswith(lowercase(x), ".png"), readdir(dir_name, join=true))
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
   
    # Use small batch size for GPU memory constraints
    batch_size = device == "GPU" ? 4 : 16
    println("Using batch size: $batch_size (auto-adjusted for $device)")
   
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
                push!(batch_ids, replace(basename(file), ".png" => ""))
            else
                failed_count += 1
            end
        end
       
        if !isempty(batch_tensors)
            try
                batch_array = cat(batch_tensors..., dims=1)
                np = pyimport("numpy")
                input_data = np.array(batch_array, dtype=np.float32)
               
                outputs = session.run([output_name], Dict(input_name => input_data))
                batch_features = outputs[1]
               
                successful_batch_count = min(length(batch_ids), size(batch_features, 1))
                
                for j in 1:successful_batch_count
                    feature_vector = vec(batch_features[j, :])
                    push!(all_features, feature_vector)
                    push!(all_ids, batch_ids[j])
                end
                
            catch e
                # If we get a memory error, try with smaller batch
                if occursin("Failed to allocate memory", string(e)) || occursin("CUDA out of memory", string(e))
                    println("\n⚠️ Memory error detected. Reducing batch size and retrying...")
                    
                    # Try with half the batch size
                    if length(batch_tensors) > 1
                        println("Retrying with half the batch...")
                        # Process in two smaller batches
                        mid_point = ceil(Int, length(batch_tensors) / 2)
                        
                        # First half
                        small_batch1 = batch_tensors[1:mid_point]
                        small_ids1 = batch_ids[1:mid_point]
                        
                        if !isempty(small_batch1)
                            batch_array1 = cat(small_batch1..., dims=1)
                            input_data1 = np.array(batch_array1, dtype=np.float32)
                            outputs1 = session.run([output_name], Dict(input_name => input_data1))
                            batch_features1 = outputs1[1]
                            
                            for j in 1:min(length(small_ids1), size(batch_features1, 1))
                                feature_vector = vec(batch_features1[j, :])
                                push!(all_features, feature_vector)
                                push!(all_ids, small_ids1[j])
                            end
                        end
                        
                        # Second half if needed
                        if length(batch_tensors) > mid_point
                            small_batch2 = batch_tensors[mid_point+1:end]
                            small_ids2 = batch_ids[mid_point+1:end]
                            
                            if !isempty(small_batch2)
                                batch_array2 = cat(small_batch2..., dims=1)
                                input_data2 = np.array(batch_array2, dtype=np.float32)
                                outputs2 = session.run([output_name], Dict(input_name => input_data2))
                                batch_features2 = outputs2[1]
                                
                                for j in 1:min(length(small_ids2), size(batch_features2, 1))
                                    feature_vector = vec(batch_features2[j, :])
                                    push!(all_features, feature_vector)
                                    push!(all_ids, small_ids2[j])
                                end
                            end
                        end
                        
                        println("✓ Successfully processed batch with reduced size")
                    end
                else
                    # Re-throw if it's not a memory error
                    rethrow(e)
                end
            end
        end
       
        ProgressMeter.update!(p, ceil(Int, batch_end / batch_size))
        
        # Force garbage collection periodically to free memory
        if i % (batch_size * 10) == 0
            GC.gc()
        end
    end
   
    if !isempty(all_features)
        df = DataFrame()
        df[!, "image_id"] = all_ids
        
        feature_dim = length(all_features[1])
        println("Feature dimension: $feature_dim")
        println("Number of images processed: $(length(all_features))")
        
        # Create all feature columns first
        for i in 1:feature_dim
            df[!, "feature_$i"] = Vector{Float32}(undef, length(all_features))
        end
        
        # Now fill the data in chunks to avoid memory issues
        chunk_size = 1000
        total_images_processed = length(all_features)
        
        for i in 1:feature_dim
            chunk_start = 1
            while chunk_start <= total_images_processed
                chunk_end = min(chunk_start + chunk_size - 1, total_images_processed)
                chunk_indices = chunk_start:chunk_end
                
                # Extract feature values for this chunk
                feature_values = [f[i] for f in all_features[chunk_indices]]
                df[chunk_indices, "feature_$i"] = feature_values
                
                chunk_start = chunk_end + 1
            end
            
            # Show progress for feature extraction
            if i % 100 == 0
                println("Processed $i/$feature_dim features...")
            end
        end
       
        elapsed = time() - start_time
        speed = length(all_features) / elapsed
       
        println("\n✅ Extraction complete: $dir_name")
        println(" Processed: $(length(all_features)) images")
        println(" Failed: $failed_count images")
        println(" Time: $(round(elapsed, digits=1)) seconds")
        println(" Speed: $(round(speed, digits=1)) img/s")
        println(" Device: $device")
       
        return df, length(all_features)
    else
        println("❌ No features extracted from $dir_name")
        return DataFrame(), 0
    end
end



function save_features(df::DataFrame, filename::String)
    if nrow(df) > 0
        # Create directory if it doesn't exist
        dir = dirname(filename)
        if !isempty(dir) && !isdir(dir)
            mkpath(dir)
            println("📁 Created directory: $dir")
        end
        
        CSV.write(filename, df)
        size_mb = round(filesize(filename) / 1024^2, digits=2)
        println("💾 Saved: $filename ($size_mb MB)")
    end
end






function main()
    println("Starting Ocean Internal Wave Feature Extraction...")
   
    if !isfile("Data/transformer_model.onnx")
        println("❌ Error: transformer_model.onnx not found!")
        return
    end
   
    has_gpu = check_hardware()
    session_info = create_session()
   
    while true
        print_menu()
        choice = get_user_choice()
       
        if choice == 4
            println("\n👋 Exiting...")
            break
        end
       
        max_images = get_max_images()
       
        if choice == 1
            println("\n📊 MODE: Extract from TRAIN images")
            df, count = extract_from_directory("Data/train", session_info, max_images)
            if count > 0
                save_features(df, "Data/features_train.csv")
            else
                println("⚠️ No features extracted from train directory")
            end
           
        elseif choice == 2
            println("\n📊 MODE: Extract from TEST images")
            df, count = extract_from_directory("Data/test", session_info, max_images)
            if count > 0
                save_features(df, "Data/features_test.csv")
            else
                println("⚠️ No features extracted from test directory")
            end
           
        elseif choice == 3
            println("\n📊 MODE: Extract from BOTH directories")
           
            println("\n" * "="^40)
            df_train, count_train = extract_from_directory("Data/train", session_info, max_images)
            if count_train > 0
                save_features(df_train, "Data/features_train.csv")
            else
                println("⚠️ No features extracted from train directory")
            end
           
            println("\n" * "="^40)
            df_test, count_test = extract_from_directory("Data/test", session_info, max_images)
            if count_test > 0
                save_features(df_test, "Data/features_test.csv")
            else
                println("⚠️ No features extracted from test directory")
            end
           
            if count_train > 0 && count_test > 0
                println("\n🔗 Combining train and test features...")
                df_all = vcat(df_train, df_test)
                save_features(df_all, "Data/features_all.csv")
                println("✅ Combined features saved: features_all.csv")
            elseif count_train > 0 || count_test > 0
                println("⚠️ Only one directory had features - skipping combined file")
            else
                println("⚠️ No features extracted from either directory")
            end
           
            println("\n📈 SUMMARY:")
            println(" Train images: $count_train")
            println(" Test images: $count_test")
            println(" Total images: $(count_train + count_test)")
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
