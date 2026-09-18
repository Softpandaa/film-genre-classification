"""Every path, column name and hyperparameter used anywhere in the project.

Nothing outside this file hard-codes a path or a training constant.
"""

import os

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw Letterboxd download from Kaggle. See README for the one manual step.
DATA_DIR = os.path.join(ROOT_DIR, "data")
POSTER_DIR = os.path.join(DATA_DIR, "posters")

MOVIES_CSV = os.path.join(DATA_DIR, "movies.csv")
GENRES_CSV = os.path.join(DATA_DIR, "genres.csv")
COUNTRIES_CSV = os.path.join(DATA_DIR, "countries.csv")
LANGUAGES_CSV = os.path.join(DATA_DIR, "languages.csv")
RELEASES_CSV = os.path.join(DATA_DIR, "releases.csv")

# The corpus of record, committed to the repo, compressed. These four tables
# are FROZEN: they are the ones every reported result was produced on. The
# Letterboxd dataset on Kaggle has been revised since, so rebuilding them from
# today's download gives a different corpus and reproduces nothing. See
# preprocess.verify_corpus.
POSTER_SIZES_CSV = os.path.join(DATA_DIR, "poster_sizes.csv.gz")
VALID_MOVIES_CSV = os.path.join(DATA_DIR, "valid_movies.csv.gz")
ENCODED_GENRES_CSV = os.path.join(DATA_DIR, "encoded_genres.csv.gz")
CNN_GENRES_CSV = os.path.join(DATA_DIR, "cnn_genres.csv.gz")

# Row count and a hash of the movie id index, per frozen table.
CORPUS_FINGERPRINT = {
    "encoded_genres": (548835, "9e426996e774c1248d2d5ef1b068c2b5"),
    "valid_movies": (552531, "306e522846993491c3f8dc5649f1c3f0"),
    "poster_sizes": (557539, "5c70673a8fe01844a5bd15ad4aebbe43"),
    "cnn_genres": (424102, "ae883e430d26ca6f3d7dff18667f981a"),
}

# Outputs. Both are gitignored and created on first run.
FIG_DIR = os.path.join(ROOT_DIR, "figures")
ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts")

# ----------------------------------------------------------------------------
# Column names
# ----------------------------------------------------------------------------

ID = "id"
MOVIE_ID = "movie_id"
GENRE = "genre"
GENRE_COUNT = "genre_count"
SIZE = "size"
DATE = "date"
RATING = "rating"
COUNTRY = "country"
LANGUAGE = "language"

# The language table carries one row per language per movie, under two types.
# The data section pools both, as the original analysis pooled them.
LANGUAGE_TYPES = ["Primary language", "Language"]

# ----------------------------------------------------------------------------
# Data definition
# ----------------------------------------------------------------------------

# The modal poster shape. Movies whose poster differs are dropped by
# preprocess.build_valid_movies.
POSTER_SHAPE = (345, 230, 3)

# One-hot genre columns carry a leading underscore, a side effect of
# pd.get_dummies(prefix="") in preprocess.build_encoded_genres.
GENRE_PREFIX = "_"
N_GENRES = 19

# The report runs two different three-genre tasks and pools them in one
# conclusion table. Both are kept because both are reported.
# Section 4: VGG16, AlexNet, both ResNets, both YOLO models.
THREE_GENRES_TRANSFER = ["Action", "Romance", "Thriller"]
# Section 5 Case 2: CLIP, EfficientNetV2, Fast-ViT.
THREE_GENRES_ADVANCED = ["Comedy", "Documentary", "Drama"]

# Shared train and test split. Every model that reports a test metric uses it.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# torchvision and timm input size, with ImageNet normalization.
IMG_SIZE = (224, 224)
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# ----------------------------------------------------------------------------
# Training, by model. Sourced from the notebook each model was run from.
# ----------------------------------------------------------------------------

BATCH_SIZE = 32
NUM_WORKERS = 6
EPOCHS = 5

# src/models/alexnet.py, src/models/resnet.py. Adam unless stated.
TORCHVISION_LR = {
    "alexnet": 1e-6,
    "resnet18": 5e-5,
    "resnet50": 1e-5,
}
# The SGD comparison run, one learning rate for all three backbones.
SGD_LR = 1e-4

# src/models/cnn.py, the scratch CNN on three-genre subsets.
CNN_SAMPLE_SEED = 4749
CNN_SPLIT_SEED = 4748
CNN_SAMPLES_PER_GENRE = 3000
# Section 3.1 calls this pool nine genres and then lists eight. Eight is what
# the code uses.
CNN_GENRE_POOL = [
    "Action",
    "Animation",
    "Comedy",
    "Documentary",
    "Drama",
    "Horror",
    "Music",
    "Romance",
]
# Indices into itertools.combinations(CNN_GENRE_POOL, 3), the three triples the
# report carries: Action Drama Romance, Comedy Drama Romance, Drama Horror Music.
CNN_REPORTED_TRIPLES = [17, 42, 52]
# Decade indicators appended to the flattened convolutional features.
CNN_DECADES = [1970, 1980, 1990, 2000, 2010, 2020]

# src/models/vgg16.py
VGG16_SEED = 4012
VGG16_FIT_SEED = 4013
VGG16_SAMPLES_PER_GENRE = 4000
VGG16_YEARS = (2000, 2024)
VGG16_LR = 1e-3
VGG16_EPOCHS = 50
VGG16_DENSE_UNITS = 128
VGG16_DROPOUT = 0.25
VGG16_MODEL_FILE = os.path.join(ARTIFACT_DIR, "CNN_vgg16.keras")
# Corner test keeps the centre of the poster and blanks a border this wide.
VGG16_CORNER_PAD = 30

# src/models/yolo.py
YOLO_WEIGHTS = "yolov8n-cls.pt"
YOLO_MULTILABEL = {"epochs": 5, "lr0": 1e-4}
YOLO_SINGLECLASS = {"epochs": 10, "lr0": 1e-3}
YOLO_DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset")
YOLO_SINGLECLASS_DIR = os.path.join(DATA_DIR, "yolo_dataset_singlecls")

# src/models/timm_models.py, run on Kaggle with a GPU.
TIMM_BATCH_SIZE = {"efficientnet": 1024, "fastvit": 512}
TIMM_LR = 1e-3
TIMM_EPOCHS = {"efficientnet": 10, "fastvit": 10}
FASTVIT_ARCH = "fastvit_t12.apple_in1k"
YEAR_EMBEDDING_DIM = 4

# src/models/clip.py
# The four encoders the report tables carry.
CLIP_ZERO_SHOT_BACKBONES = ["RN50", "RN50x4", "ViT-B/32", "ViT-B/16"]
