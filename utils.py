from PIL import Image
import json
import logging
import os
import warnings
import yaml

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

root_path = os.path.dirname(
    os.path.realpath(
        __file__
    )
)

config_path = os.path.join(
    root_path,
    "config.yaml"
)

font_path = os.path.join(
    root_path,
    "font.ttf"
)

outcome_path = os.path.join(
    root_path,
    "outcome"
)

checkpoint_path = os.path.join(
    outcome_path,
    "checkpoint.pt"
)

release_path = os.path.join(
    outcome_path,
    "release.pt"
)

archive_path = os.path.join(
    outcome_path,
    "archive.json"
)

demo_path = os.path.join(
    outcome_path,
    "demo"
)


def load_json(path):
    with open(path, "rt") as stream:
        buffer = json.load(stream)

    return buffer


def dump_json(
        buffer,
        path
):
    with open(path, "wt") as stream:
        json.dump(
            buffer,
            stream
        )


def load_yaml(path):
    with open(path, "rt") as stream:
        buffer = yaml.safe_load(stream)

    return buffer


def load_image(path):
    with Image.open(path) as stream:
        buffer = stream.copy()
        buffer.path = path

    return buffer
