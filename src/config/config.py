"""Config file with all the paths needed."""

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
CONFIG_PATH = PROJ_ROOT/ "src" / "config" / "yolov3_ts_train.cfg"
WEIGHTS_PATH = PROJ_ROOT/ "src" / "models" / "ts_model.pt"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_DIR_TS = RAW_DATA_DIR / "ts"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

METRICS_DIR = PROJ_ROOT / "src/metrics"

MODELS_DIR = PROJ_ROOT / "src/models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PARAMS_PATH = PROJ_ROOT/ "params.yaml"

TRAIN_IMAGES_PATH = PROCESSED_DATA_DIR / "train_images.txt"
TEST_IMAGES_PATH = PROCESSED_DATA_DIR / "test_images.txt"

SETTINGS_PATH = PROJ_ROOT / "settings.json"

YAML_FILE = PROJ_ROOT/ "dataset.yaml"

MLFLOW_FILE = PROJ_ROOT/ "runs"


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
