# 🌊 Ocean Internal Waves Detection - Hackathon Solution

## Download Data files

<https://s3.aircentre.org/public/julia_eo_26_hackathon_files.zip>

Unzip and place inside `Data/`.

**Vision Transformer Features + XGBoost**
_Final submission ready_

🚨 The Key Bug & Fix

Feature extraction produced IDs in local order (`100028`, `100047`...), but `train.csv`/`test.csv` use competition IDs (`603303.png`, ...).

→ **0% match** → model would fail silently.

**Solution**: `final_correct_fix.jl` realigns features to exact competition order.

🔄 Correct Execution Order (MUST follow exactly)

1. Python setup
   python Python_Scripts/export_transformer.py

2. Julia pipeline
   julia Scripts/Extraction/extract_features.jl  
   julia Reference_Solutions/final_correct_fix.jl
   julia Scripts/Evaluation/auc_roc_cv.jl  
   julia Scripts/Training/train_final.jl
   julia Scripts/Evaluation/predict.jl

## Submitting your solution

Please submit your solution by following the instructions bellow:

1. Fork this repo
2. Add your solution to `machine_learning_challenge_iws_solution_submissions/**your_name**`, and commit.
3. Open a Pull Request

Your submission should contain:

- The code to reproduce your solution
- `README.md` containing precise instructions on how to set up the environment and run the code
- The final `submission.csv` output file with the relevant performance metrics (kaggle compatible)
