"""Paths and settings for the FindYourFit v2 offline pipeline (H&M dataset)."""
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- raw data (Kaggle H&M)
RAW_DIR = ROOT / "data" / "raw" / "hm"
ARTICLES_CSV = RAW_DIR / "articles.csv"
TRANSACTIONS_CSV = RAW_DIR / "transactions_train.csv"
RAW_IMAGES_DIR = RAW_DIR / "images"            # images/<first 3 digits>/<10-digit id>.jpg

# ---------------------------------------------------------------- outputs
PROCESSED_DIR = ROOT / "data" / "processed"
CATALOG = PROCESSED_DIR / "catalog.parquet"
TRANSACTIONS = PROCESSED_DIR / "transactions.parquet"
PREPARE_REPORT = PROCESSED_DIR / "prepare_report.json"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"   # <encoder>_image.npy / <encoder>_text.npy
TWO_TOWER_DIR = PROCESSED_DIR / "two_tower"
IMAGES_DIR = ROOT / "data" / "images"           # web-sized product images served to the app
MODELS_DIR = ROOT / "backend" / "models"        # serving bundle loaded by Flask
REPORTS_DIR = ROOT / "reports"

# ---------------------------------------------------------------- catalog rules
GARMENT_GROUPS = ["Garment Upper body", "Garment Lower body", "Garment Full body"]
EXCLUDED_INDEX_GROUPS = ["Baby/Children"]
ASSORTMENT_WEEKS = 12     # "current assortment" = sold within the last N weeks of data
MIN_SALES = 3             # ... at least this many times
MAX_ITEMS = 30_000        # cap, most-sold first

# ---------------------------------------------------------------- interaction window + split
HISTORY_WEEKS = 26        # purchases used for training / evaluation
LAST_DAY = date(2020, 9, 22)   # last day in transactions_train.csv
VAL_START = date(2020, 9, 9)   # val  = 2020-09-09 .. 2020-09-15
TEST_START = date(2020, 9, 16)  # test = 2020-09-16 .. 2020-09-22 (last week, as in the Kaggle competition)

# ---------------------------------------------------------------- images
WEB_IMAGE_MAX_SIDE = 600  # stored copies used by the app and by the encoders
ENCODER_IMAGE_SIZE = 224

# ---------------------------------------------------------------- encoders
ENCODERS = {
    "fashion_clip": "patrickjohncyh/fashion-clip",
    "clip": "openai/clip-vit-base-patch32",
    "vgg16": "torchvision/vgg16-imagenet",
}
PRIMARY_ENCODER = "fashion_clip"

SEED = 42


def image_path(article_id: int) -> Path:
    """Web-sized image written by prepare_hm.py."""
    return IMAGES_DIR / f"{article_id:010d}.jpg"


def raw_image_path(article_id: int) -> Path:
    aid = f"{article_id:010d}"
    return RAW_IMAGES_DIR / aid[:3] / f"{aid}.jpg"
