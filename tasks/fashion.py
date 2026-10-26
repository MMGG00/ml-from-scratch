import gzip
import os
import urllib.request

import numpy as np

# Fashion-MNIST: 70,000 real grayscale photos of clothing, 28 x 28 pixels, from the online store Zalando.
# 10 kinds of item; position in this list = the label number the models see  [label 9 = "ankle boot"]
LABELS = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FOLDER = os.path.join("data", "fashion")       # where we keep our downloaded copies
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",    # 60,000 training photos + their labels
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]      # 10,000 test photos + their labels


def read_images(path: str) -> np.ndarray:
    # The file is gzip-compressed (like a zip). Inside: a 16-byte header (a label saying what's in the file),
    # then one byte per pixel: 0 = black ... 255 = white.
    raw = gzip.open(path).read()                          # bytes  [train: 47,040,016 of them]
    pixels = np.frombuffer(raw, dtype=np.uint8, offset=16)  # skip the 16 header bytes; each byte = 1 whole number 0-255
    return pixels.reshape(-1, 28 * 28) / 255             # one row of 784 pixels per photo, scaled to 0-1  [(60000, 784)]


def read_labels(path: str) -> np.ndarray:
    # Same idea, but the header is 8 bytes and then one byte per label (0-9).
    raw = gzip.open(path).read()                          # bytes  [train: 60,008 of them]
    return np.frombuffer(raw, dtype=np.uint8, offset=8).astype(int)   # [9, 0, 0, 3, 0, 2, ...] = boot, t-shirt, t-shirt, dress ...


# Model knobs this task wants changed from their defaults:  {model kind: {knob name: value}}.
# logistic: learning rate 0.1 instead of 0.5. Each photo has hundreds of non-zero pixels, so the slopes are big.
# Tested: 0.826 vs 0.812.
# cnn: the photos are 28 x 28.
SETTINGS = {"logistic": {"learning_rate": 0.1}, "cnn": {"side": 28}}

def load(train_size: int = 6000, test_size: int = 2000) -> tuple:
    os.makedirs(FOLDER, exist_ok=True)
    for name in FILES:                                    # download each file once; after that, read the saved copy
        path = os.path.join(FOLDER, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(URL + name, path)

    train_x = read_images(os.path.join(FOLDER, FILES[0]))[:train_size]   # (6000, 784)
    train_y = read_labels(os.path.join(FOLDER, FILES[1]))[:train_size]   # (6000,)
    test_x = read_images(os.path.join(FOLDER, FILES[2]))[:test_size]     # (2000, 784)
    test_y = read_labels(os.path.join(FOLDER, FILES[3]))[:test_size]     # (2000,)
    return train_x, train_y, test_x, test_y