# MobileNetV3 + CBAM Research Project

## Project Overview
This project is a research implementation of **MobileNetV3 (Large and Small)** integrated with **CBAM (Convolutional Block Attention Module)** and **SE (Squeeze-and-Excitation)** attention mechanisms for image classification. It is designed to evaluate the performance and complexity trade-offs of different attention strategies on plant disease datasets.

### Core Architecture
- **`src/models/`**: Unified model definitions supporting parameterizable attention.
- **`src/engine.py`**: Standardized training and validation loops with early stopping and logging.
- **`src/config.py`**: Centralized path management (Data, Results, Checkpoints).
- **`scripts/`**: CLI-based entry points for major tasks.

## Building and Running

### Environment Setup
The project uses Conda for environment management and `pip` for dependencies.
- **Conda Environment**: `comvis` (Python 3.11)
- **Install Dependencies**: `pip install -r requirements.txt`

### Key Commands
- **Train a model**:
  ```bash
  python scripts/train.py --model <model_name> --epochs <num> --data_dir <path_to_data>
  ```
  Available models: `mobilenetv3_small`, `mobilenetv3_large`, `proposed_large_16`, `proposed_large_32`, `proposed_small_16`, `proposed_small_32`, `mobilenetv2`, `shufflenetv2`.
- **Evaluate a model**:
  ```bash
  python scripts/eval.py --model <model_name> --weight <checkpoint_filename> --data_dir <path_to_data>
  ```
  *Note: Evaluation now automatically prints hardware specifications and calculates Latency/Throughput.*
- **Measure Complexity (FLOPs/Params/Memory)**:
  ```bash
  python scripts/measure_complexity.py
  ```
  *Note: Memory size is calculated based on float32 parameters (4 bytes per parameter).*

## Development Conventions

### Path Management
- **NEVER** use hardcoded strings for file paths.
- Always import path constants from `src.config` (e.g., `from src.config import DATA_DIR, RESULTS_DIR`).
- Use `os.path.join` for cross-platform compatibility.

### Imports
- Use absolute imports starting from the `src` package for internal modules (e.g., `from src.utils import ...`).
- Entry scripts in `scripts/` must include the project root in `sys.path` to ensure absolute imports work correctly.

### Model Modification
- When adding new attention modules or model variants, update the `model_map` in `scripts/train.py`, `scripts/eval.py`, and `scripts/measure_complexity.py` to ensure consistency.

### Logging and Results
- Training results are saved in `results/training/`.
- Checkpoints are saved in `results/checkpoints/`.
- Evaluation dashboards and metrics are saved in `results/evaluation/`.
Analy