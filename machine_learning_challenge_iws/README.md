# 🌊 Ocean Internal Waves Detection - Hackathon Solution

## Download Data files
https://s3.aircentre.org/public/julia_eo_26_hackathon_files.zip

Unzip and place inside `Data/`.

**Vision Transformer Features + XGBoost**
*Final submission ready*

 🚨 The Key Bug & Fix

Feature extraction produced IDs in local order (`100028`, `100047`...), but `train.csv`/`test.csv` use competition IDs (`603303.png`, ...).  

→ **0% match** → model would fail silently.

**Solution**: `final_correct_fix.jl` realigns features to exact competition order.

 🔄 Correct Execution Order (MUST follow exactly)

```bash
cd Julia_EO_Hackathon_2026

1. Extract features (if not already done)
python Python_Scripts/export_transformer.py

2. CRITICAL: Align features to competition IDs
julia Reference_Solutions/final_correct_fix.jl
 → Creates features_*_CORRECTLY_ALIGNED.csv

3. Train final model
julia Scripts/Training/train_final.jl

4. Generate submission
julia Scripts/Evaluation/predict.jl
 → Creates Data/final_submission.csv (upload this!)
