# ADS CPU preparation tools

Python >=3.8, standard library only. No pip, GPU, model calls, or training.

Run from the scale directory:

    python -m unittest discover -s control/scale-ads-tools -p "test_*.py" -v
    python control/scale-ads-tools/ads_tools.py prepare --root "$PWD"

Protocol: ADS_PUBLIC_ADAPTED_V1. Seed 42 and ~20% validation are our adaptation choices. Original train is split by question and exact-answer groups. Expected rows: train 885, validation 222, test 475. Original train/test files are unchanged. Cross-test duplicates are preserved and reported.

Inputs and teacher labels are in separate files in data/prepared/ads_public_adapted_v1. The script verifies source checksums against reports/data_audit.json. Existing outputs with different contents are never overwritten.

Evaluation (only after real predictions exist): use ads_tools.py evaluate --help. Prediction JSONL requires sample_id and prediction_raw. Raw scores are 0..10, normalized error divides by the fixed max 10. QWK uses integer gold grades and floor(prediction+0.5), with out-of-range predictions rejected. These metric choices are adapted, not verified author choices. Undefined metrics are null. Predictions are joined by ID, never row order. Test fixtures in unittest are synthetic and are not model results.

Local verification: six unittest cases, full dataset preparation, repeat execution, and Python 3.8 syntax checks passed under Python 3.11. Actual server Python 3.8 execution still needs the above commands.
