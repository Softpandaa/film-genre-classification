# A Comparison of Neural Architectures for Movie Poster Genre Classification

A movie poster is the first visual contact a viewer has with a film, and it signals genre before any other information reaches the audience. We measure how far that signal survives as a machine-readable one, and at what cost. Three regimes are compared, a convolutional network, four ImageNet backbones transferred to posters, and a vision-language model, against two backbones built for cheap inference, EfficientNetV2 and Fast-ViT. 

## Findings

We found that the classifier draws on the whole poster rather than its center, tolerates rotation and relies mainly on color for edge detection. Poster design changes measurably across decades, and supplying the decade as an indicator raises the accuracy by about 4%. ResNets deliver the strongest results, reaching 54.5% accuracy on the 19-genre task and 69.7% on the simplified 3-genre task.

## Layout

    data/        the frozen dataset, and the Kaggle download when present
    src/         config.py, the shared modules, models/ one file per family
    report.pdf   

## Data

The project uses the Letterboxd dataset on Kaggle,
<https://www.kaggle.com/datasets/gsimonx37/letterboxd>, namely `genres.csv`,
`movies.csv`, `countries.csv`, `languages.csv`, `releases.csv` and
`posters.zip`. Cleaning keeps the posters of the modal shape and caps the genre
count at four, which leaves 548,835 movies in the four tables committed to
`data/`.

| File | Rows | Contents |
|---|---|---|
| `encoded_genres.csv.gz` | 548,835 | one-hot genres per movie, with a genre count |
| `valid_movies.csv.gz` | 552,531 | movies whose poster has the modal shape |
| `poster_sizes.csv.gz` | 557,539 | dimensions of every readable poster |
| `cnn_genres.csv.gz` | 424,102 | post-1970 subset with release dates |

The train and test split is positional over the row order of `encoded_genres.csv.gz`. To train or predict, the posters are needed. Download `posters.zip` and extract it so that images sit at `data/posters/<movie_id>.jpg`. 

## Reproducing

    pip install -r requirements.txt
    pip install git+https://github.com/openai/CLIP.git
    python src/run.py <command>

| Command | Produces |
|---|---|
| `verify` | the check of the four committed tables |
| `eda` | the poster size table and the five data figures |
| `cnn` | the six settings of the network built from scratch |
| `vgg16` | the VGG16 classifier and its three inspection tests |
| `torchvision` | AlexNet, ResNet-18 and ResNet-50 |
| `yolo` | YOLOv8 under both label conventions |
| `timm` | EfficientNetV2 and Fast-ViT |
| `clip` | the four zero-shot encoders |
| `tables` | the two comparison tables and the per-genre AUROC |
| `posters` | the decade poster strips |
