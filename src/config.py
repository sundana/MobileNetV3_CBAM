import os

# Project Root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Output Directories
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
TRAINING_LOG_DIR = os.path.join(RESULTS_DIR, "training")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
CSV_DIR = os.path.join(RESULTS_DIR, "csv")
EVAL_DIR = os.path.join(RESULTS_DIR, "evaluation")
COMPLEXITY_DIR = os.path.join(RESULTS_DIR, "complexity")
CONFUSION_MATRIX_DIR = os.path.join(RESULTS_DIR, "confusion_matrix")

# Create directories if they don't exist
for d in [RESULTS_DIR, CHECKPOINT_DIR, TRAINING_LOG_DIR, PLOT_DIR, CSV_DIR, EVAL_DIR, COMPLEXITY_DIR, CONFUSION_MATRIX_DIR]:
    os.makedirs(d, exist_ok=True)
