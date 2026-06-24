# Fanti Driving

## Setup Environment

0. Requirements:
   - Python >= `3.13`

1. Create virtual environment:

   ```bash
   python -m venv .venv
   ```

2. Enter the virtual environment:

   ```bash
   source .venv/bin/activate # Linux
   .\.venv\Scripts\Activate.ps1 # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

If you are using PowerShell, please run the following command before executing following commands:

```ps1
$env:PYTHONPATH = "."
```

### Feature Extraction

1. Install dataset from [Kaggle](https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset)
2. Place all videos under `data/` directory like:

   ```text
   data/
   ├── Fold1_part1/
   │   ├── 01/
   │   │   ├── 0.mov
   │   │   ├── 5.mov
   │   │   └── 10.MOV
   │   ├── 02/
   │   │   ├── 0.mov
   │   │   ├── 5.MOV
   │   │   └── 10.MOV
   │   ├── 03/
   │   │   ├── 0.MOV
   │   │   ├── 5.mov
   │   │   └── 10.mov
   │   └── ...
   ├── Fold1_part2/
   └── ...
   ```

3. Modify `DATA_DIR` and `OUTPUT_PATH` in `scripts/build_csv.py` (Optional)
4. Build CSV file which contains videos to be processed

   ```bash
   python scripts/build_csv.py
   ```

5. In `scripts/dataset.py`, set `DATASET_LIST_PATH` to `OUTPUT_PATH` from step 3, and optionally set `RAW_FEATURE_PATH` and `ENHANCED_FEATURE_PATH`
6. Generate dataset

   ```bash
   python scripts/dataset.py
   ```

7. Feature dataset can be found in `ENHANCED_FEATURE_PATH`

### Train Model

1. Set `PATH` in `scripts/train.py` to `ENHANCED_FEATURE_PATH`
2. Train the model

   ```bash
   python scripts/train.py
   ```

3. The model can be found in `models/fatigue_model.pkl`

### Run the Application

1. Modify the `MODEL_PATH` in `scripts/app.py` to the model path (Optional)
2. Start the application

   ```bash
   python scripts/app.py
   ```

3. Access [http://localhost:5000](http://localhost:5000) to use the application

## Project Architecture

```bash
.
├── models
│   └── heuristic_model.pkl     # Model file
├── scripts/                    # Entry points
├── src
│   ├── dataset/                # Dataset operations
│   ├── features/               # Feature engineering
│   ├── preprocessing/          # Preprocessing tools
│   ├── realtime/               # Realtime prediciton
│   ├── train/                  # Model training
│   └── visualization/          # Figure generation scripts
├── static
│   ├── css/                    # Stylesheet
│   └── js/                     # Frontend logic
├── templates
│   └── index.html              # Frontend structure
├── .editorconfig
├── .envrc
├── .gitattributes
├── .gitignore
├── .prettierrc
├── face_landmarker.task        # Face recognition model
├── flake.lock
├── flake.nix
├── README.md
└── requirements.txt            # Dependencies
```
