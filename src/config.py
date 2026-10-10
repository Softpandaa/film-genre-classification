"""Paths, column names and model settings."""

import os

# Paths -----------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Raw Letterboxd download from Kaggle, see README.
POSTER_DIR = os.path.join(DATA_DIR, "posters")
MOVIES_CSV = os.path.join(DATA_DIR, "movies.csv")
GENRES_CSV = os.path.join(DATA_DIR, "genres.csv")
COUNTRIES_CSV = os.path.join(DATA_DIR, "countries.csv")
LANGUAGES_CSV = os.path.join(DATA_DIR, "languages.csv")
RELEASES_CSV = os.path.join(DATA_DIR, "releases.csv")

# Frozen corpus of every reported result, in the repo. Kaggle has revised the
# dataset since, so a rebuild from today's download gives different tables.
POSTER_SIZES_CSV = os.path.join(DATA_DIR, "poster_sizes.csv.gz")
VALID_MOVIES_CSV = os.path.join(DATA_DIR, "valid_movies.csv.gz")
ENCODED_GENRES_CSV = os.path.join(DATA_DIR, "encoded_genres.csv.gz")
CNN_GENRES_CSV = os.path.join(DATA_DIR, "cnn_genres.csv.gz")
# Rows and MD5 of the index column, per table.
CORPUS_FINGERPRINT = {
    "encoded_genres": (548835, "9e426996e774c1248d2d5ef1b068c2b5"),
    "valid_movies": (552531, "306e522846993491c3f8dc5649f1c3f0"),
    "poster_sizes": (557539, "5c70673a8fe01844a5bd15ad4aebbe43"),
    "cnn_genres": (424102, "ae883e430d26ca6f3d7dff18667f981a"),
}

# Outputs, created on first run.
FIG_DIR = os.path.join(ROOT_DIR, "figures")
ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts")

# Columns ---------------------------------------------------------------------
ID = "id"
MOVIE_ID = "movie_id"
GENRE = "genre"
GENRE_COUNT = "genre_count"
SIZE = "size"
DATE = "date"
RATING = "rating"
COUNTRY = "country"
LANGUAGE = "language"
LANGUAGE_TYPES = ["Primary language", "Language"]  # row types, pooled in Figure 2

# Data ------------------------------------------------------------------------
POSTER_SHAPE = (345, 230, 3)  # modal shape, the one Section 2 keeps
GENRE_PREFIX = "_"  # one-hot columns read "_<genre>", from pd.get_dummies(prefix="")
N_GENRES = 19

# Two different three-genre tasks, both reported and pooled in Table 17.
THREE_GENRES_TRANSFER = ["Action", "Romance", "Thriller"]  # Section 4
THREE_GENRES_ADVANCED = ["Comedy", "Documentary", "Drama"]  # Section 5

# Shared train and test split, by row position in encoded_genres.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# torchvision and timm input size, with ImageNet normalization.
IMG_SIZE = (224, 224)
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# Training --------------------------------------------------------------------
BATCH_SIZE = 32
NUM_WORKERS = 6
EPOCHS = 5

# AlexNet and ResNet ----------------------------------------------------------
TORCHVISION_LR = {"alexnet": 1e-6, "resnet18": 5e-5, "resnet50": 1e-5}  # Adam
SGD_LR = 1e-4  # the SGD runs of Figure 9, all three backbones

# Scratch CNN -----------------------------------------------------------------
CNN_SAMPLE_SEED = 4749
CNN_SPLIT_SEED = 4748
CNN_SAMPLES_PER_GENRE = 3000
CNN_GENRE_POOL = ["Action", "Animation", "Comedy", "Documentary",
                  "Drama", "Horror", "Music", "Romance"]
# The triples of Table 1, as indices into itertools.combinations(CNN_GENRE_POOL, 3).
CNN_REPORTED_TRIPLES = [17, 42, 52]
CNN_DECADES = [1970, 1980, 1990, 2000, 2010, 2020]  # decade indicators of Section 3.1

# VGG16 -----------------------------------------------------------------------
VGG16_SEED = 4012
VGG16_FIT_SEED = 4013
VGG16_SAMPLES_PER_GENRE = 4000
VGG16_YEARS = (2000, 2024)  # release window, inclusive
VGG16_LR = 1e-3
VGG16_EPOCHS = 50
VGG16_DENSE_UNITS = 128
VGG16_DROPOUT = 0.25
VGG16_MODEL_FILE = os.path.join(ARTIFACT_DIR, "CNN_vgg16.keras")
VGG16_CORNER_PAD = 30  # pixels blanked on each side in the border test of Table 4

# YOLOv8 ----------------------------------------------------------------------
YOLO_WEIGHTS = "yolov8n-cls.pt"
YOLO_MULTILABEL = {"epochs": 5, "lr0": 1e-4}
YOLO_SINGLECLASS = {"epochs": 10, "lr0": 1e-3}  # the multi-class approach
YOLO_DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset")
YOLO_SINGLECLASS_DIR = os.path.join(DATA_DIR, "yolo_dataset_singlecls")

# EfficientNetV2 and Fast-ViT -------------------------------------------------
TIMM_BATCH_SIZE = {"efficientnet": 1024, "fastvit": 512}
TIMM_LR = 1e-3
TIMM_EPOCHS = {"efficientnet": 10, "fastvit": 10}
FASTVIT_ARCH = "fastvit_t12.apple_in1k"
YEAR_EMBEDDING_DIM = 4

# CLIP ------------------------------------------------------------------------
CLIP_ZERO_SHOT_BACKBONES = ["RN50", "RN50x4", "ViT-B/32", "ViT-B/16"]  # Tables 13 and 14
