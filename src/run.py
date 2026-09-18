"""Single entry point. One command per section of the report.

    python src/run.py verify              check the frozen corpus
    python src/run.py eda                 figures and tables of section 2
    python src/run.py posters             the appendix poster strips
    python src/run.py cnn                 section 3, six settings
    python src/run.py vgg16               section 4.1, fit and inspect
    python src/run.py torchvision         section 4.2, AlexNet and both ResNets
    python src/run.py yolo                section 4.3, both approaches
    python src/run.py timm                section 5, EfficientNetV2 and Fast-ViT
    python src/run.py clip                section 5, zero shot and fine tuning
    python src/run.py tables              the two comparison tables of section 6

Each command imports only what it needs, so a command runs even when the
frameworks of the other sections are not installed.
"""

import argparse
import os
from itertools import combinations

import config


def verify():
    from preprocess import verify_corpus
    print(verify_corpus().to_string(index=False))


def eda():
    """Section 2. The poster size table and the five figures."""
    import eda as section_two
    print(section_two.poster_size_table().to_string())
    for path in section_two.figures():
        print("wrote", path)


def posters():
    """The decade poster strips of the appendix. Needs the poster images."""
    from preprocess import sample_decade_posters
    for decade, ids in sample_decade_posters().items():
        print(decade, ids)


def cnn():
    """The three genre combinations of section 3, each under two windows."""
    from models import cnn as model
    from sklearn.metrics import roc_auc_score, f1_score, precision_score
    import numpy as np

    chosen = [list(c) for i, c in enumerate(combinations(config.CNN_GENRE_POOL, 3))
              if i in config.CNN_REPORTED_TRIPLES]
    for genres in chosen:
        for min_year in (1970, 2000):
            fitted, (x_test, y_test, year_test) = model.run(genres, min_year)
            probabilities = fitted.predict(
                [x_test, model.decade_indicators(year_test)])
            predicted = probabilities.argmax(axis=1)
            true = y_test.argmax(axis=1)
            print(genres, min_year,
                  "auroc %.6f" % roc_auc_score(y_test, probabilities,
                                               average="macro"),
                  "f1 %.6f" % f1_score(true, predicted, average="macro"),
                  "precision %.6f" % precision_score(true, predicted,
                                                     average="macro"),
                  "accuracy %.6f" % np.mean(true == predicted))


def vgg16():
    """Section 4.1, the classifier and the three inspection tests."""
    from models import vgg16 as model
    fitted, (x_test, y_test) = model.run()
    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    fitted.save(config.VGG16_MODEL_FILE)
    print(model.inspect(fitted, x_test, y_test).to_string())


def torchvision_models():
    """Section 4.2. Each model writes a prediction array for run.py tables.

    The optimiser comparison of that section is the same three builders called
    with optimizer=torch.optim.SGD and lr=config.SGD_LR.
    """
    from models import alexnet, resnet
    from models.lightning import train, predict_all
    for name, build in (("alexnet", alexnet.build),
                        ("resnet18", lambda: resnet.build("resnet18")),
                        ("resnet50", lambda: resnet.build("resnet50"))):
        print(predict_all(train(build()), name))


def yolo():
    """Section 4.3, both label conventions."""
    from models import yolo as model
    for name, root, settings, multi_label in (
            ("yolo_multilabel", config.YOLO_DATASET_DIR,
             config.YOLO_MULTILABEL, True),
            ("yolo_multiclass", config.YOLO_SINGLECLASS_DIR,
             config.YOLO_SINGLECLASS, False)):
        model.build_folders(root, multi_label=multi_label)
        print(model.predict_all(model.train(root, settings), name))


def timm_models():
    """Section 5, EfficientNetV2 and Fast-ViT, with and without the year."""
    from models import timm_models as model
    frame = model.population()
    for name, builder, batch in (
            ("efficientnet", model.EfficientNetWithYear,
             config.TIMM_BATCH_SIZE["efficientnet"]),
            ("fastvit", model.build_fastvit,
             config.TIMM_BATCH_SIZE["fastvit"])):
        train_loader, test_loader, years = model.loaders(frame, batch)
        classes = frame["label"].nunique()
        net = (builder(classes, 0) if name == "efficientnet"
               else builder(classes))
        net = model.train(net, train_loader, config.TIMM_EPOCHS[name])
        print(model.predict(net, test_loader, len(frame), frame, name))


def clip_models():
    """Section 5, the four zero-shot encoders."""
    from models import clip as model
    from data_io import genre_names, load_encoded_genres
    names = genre_names(load_encoded_genres().columns[:-1])
    for backbone in config.CLIP_ZERO_SHOT_BACKBONES:
        scores = model.zero_shot(backbone, names)
        print(model.save(scores, "clip_" + backbone.replace("/", "-")))


def tables():
    """Section 6. Reads whatever prediction arrays are present."""
    import pandas as pd
    from evaluate import load_predictions, metrics

    settings = {
        "alexnet":         dict(average="macro", auroc_on="label", in_test=True),
        "resnet18":        dict(average="macro", auroc_on="label", in_test=True),
        "resnet50":        dict(average="macro", auroc_on="label", in_test=True),
        "yolo_multilabel": dict(average="micro", auroc_on="score", in_test=False),
        "yolo_multiclass": dict(average="micro", auroc_on="score", in_test=False),
    }
    for label, genres in (("3 genres", config.THREE_GENRES_TRANSFER),
                          ("19 genres", None)):
        rows = []
        for name, kwargs in settings.items():
            path = os.path.join(config.ARTIFACT_DIR, f"{name}_predictions.npz")
            if not os.path.exists(path):
                continue
            rows.append(dict(model=name,
                             **metrics(load_predictions(path), genres=genres,
                                       **kwargs)))
        print(f"\n{label}")
        print(pd.DataFrame(rows).to_string(index=False) if rows
              else "  no prediction arrays found in " + config.ARTIFACT_DIR)

    print("\nAUROC by genre, test half")
    from data_io import split_indices
    from evaluate import auroc_by_genre
    columns = {}
    for name in ("alexnet", "resnet18", "resnet50"):
        path = os.path.join(config.ARTIFACT_DIR, f"{name}_predictions.npz")
        if not os.path.exists(path):
            continue
        scores = load_predictions(path)
        _, test_indices = split_indices(len(scores))
        by_genre = auroc_by_genre(scores, indices=test_indices)
        columns[name] = by_genre.set_index("Genre")["ROC AUC"]
    print(pd.DataFrame(columns).to_string() if columns else "  nothing to report")


COMMANDS = {"verify": verify, "eda": eda, "posters": posters,
            "cnn": cnn, "vgg16": vgg16,
            "torchvision": torchvision_models, "yolo": yolo,
            "timm": timm_models, "clip": clip_models, "tables": tables}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    COMMANDS[parser.parse_args().command]()
