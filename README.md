# Fanti Driving

## Development

0. Requirements:
   - Python >= `3.13`

1. Create virtual environment:

   ```bash
   python -m venv .venv
   ```

2. Enter the virtual environment:

   ```bash
   source .venv/bin/activate # Linux
   .\.venv\bin\Activate.ps1 # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Project Architecture

```bash
Fanti_Driving/
├── app.py
├── requirements.txt
├── face_landmarker.task
│
├── models/
│   ├── fatigue_model.pkl
│   └── heuristic_model.pkl
│
├── src/
│   │
│   ├── preprocessing/
│   │   ├── video_loader.py
│   │   └── frame_extractor.py
│   │
│   ├── features/
│   │   ├── frame_schema.py
│   │   └── window_agg.py
│   │
│   ├── train/
│   │   ├── chen_train.py
│   │   ├── heuristic_selection.py
│   │   └── heuristic_train.py
│   │
│   └── realtime/
│   │   ├── buffer.py
│       └── predictor.py
│
├── dataset/
│   ├── build_csv.py
│   ├── extract_features.py
│   ├── baseline.py
│   ├── merge_five.csv
│   ├── merge_five_enhanced.csv
│   └── merge_five_enhanced_new.csv
│
└── templates
    └── index.html
```