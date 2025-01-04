from pathlib import Path

root = Path(__file__).parent.parent  # top-level directory
DATA_PATH = root / "data/"  # datasets and pretrained weights
TRAINING_PATH = root / "outputs/training/"  # training checkpoints
TRAINING_PATH1 = "/home/weitong/code/VisualOverlap/glue-factory/outputs/training/"  # training checkpoints

EVAL_PATH = root / "outputs/results/"  # evaluation results
