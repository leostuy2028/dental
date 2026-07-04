import base64
import pandas as pd
from io import BytesIO
from PIL import Image


def load_closed(path="data/closed_ended.parquet"):
    return pd.read_parquet(path)


def load_open(path="data/open_ended.parquet"):
    return pd.read_parquet(path)


def decode_image(b64_string):
    return Image.open(BytesIO(base64.b64decode(b64_string)))
