import io
import os
import random
import urllib.request
import wave
import zipfile

import numpy as np

# Free Spoken Digit Dataset: 3000 short recordings. 6 people each say every digit 0-9, 50 times.
# Each file is named digit_speaker_take.wav  [7_theo_12.wav = theo saying "7", take 12]
URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
FILE = os.path.join("data", "spoken_digits.zip")    # where we keep our downloaded copy (about 20 MB)
TEST_SPEAKER = "nicolas"    # the voice the models never hear in training; his scores are closest to the 6-person average


def sound_to_numbers(sound: np.ndarray, slice_size: int = 256, step: int = 32,
                     bands: int = 32, chunks: int = 16) -> np.ndarray:
    # Turns one recording (any length) into a fixed row of chunks * bands numbers  [16 * 32 = 512].
    # sound: np.ndarray of floats between -1 and 1, 8000 per second  [a 0.42 s clip -> 3360 numbers]

    # 1. Cut the sound into short overlapping slices: 256 numbers each (0.032 s), a new one every 32 numbers.
    starts = range(0, len(sound) - slice_size + 1, step)               # where each slice begins  [0, 32, 64, ...]
    slices = np.array([sound[s:s + slice_size] for s in starts])       # (slices, 256)  [3360 numbers -> 98 slices]

    # 2. For each slice, measure how strong each pitch is, from low to high.
    #    np.fft.rfft does this: 256 loudness numbers in -> 129 pitch strengths out. We keep the first 128.
    pitch = np.abs(np.fft.rfft(slices, axis=1))[:, :128]                # (slices, 128)

    # 3. Group the 128 pitches into 32 bands of 4 neighbors, and average each band.
    pitch = pitch.reshape(len(slices), bands, -1).mean(axis=2)          # (slices, 32)  [98 x 32]
    pitch = np.log(1 + 100 * pitch)    # squash big values so loud sounds don't drown out quiet ones

    # 4. Squeeze time into 16 chunks (groups of neighboring slices), averaging each chunk.
    pieces = np.array_split(pitch, chunks)                              # 16 groups  [98 slices -> groups of 6 or 7]
    grid = np.array([piece.mean(axis=0) for piece in pieces])           # (16, 32): a tiny picture of the sound
    return grid.flatten()                                               # (512,): one row, like a digits image


def load() -> tuple:
    if not os.path.exists(FILE):                        # download once; after that, read the saved copy
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(URL, FILE)

    train_rows, train_labels, test_rows, test_labels = [], [], [], []
    with zipfile.ZipFile(FILE) as z:                    # read files straight out of the zip, no unzipping needed
        names = sorted(n for n in z.namelist() if n.endswith(".wav"))   # list[str], 3000 paths inside the zip
        random.seed(0)
        random.shuffle(names)    # mix the order; sorted would put all the 9s last, and the models'
                                 # check pile is the LAST 20% of training rows, so it'd be all 9s

        for name in names:       # name: str  ["free-spoken-digit-dataset-master/recordings/7_theo_12.wav"]
            digit, speaker, _ = os.path.basename(name).split("_")   # "7_theo_12.wav" -> "7", "theo", "12.wav"

            with wave.open(io.BytesIO(z.read(name))) as w:   # z.read gives the file's bytes; BytesIO lets wave read them like a file
                raw = w.readframes(w.getnframes())           # bytes: every loudness number, 2 bytes each
            sound = np.frombuffer(raw, dtype=np.int16) / 32768   # bytes -> whole numbers -32768..32767 -> floats -1..1

            row = sound_to_numbers(sound)                    # (512,)
            if speaker == TEST_SPEAKER:                      # nicolas -> test pile, everyone else -> training pile
                test_rows.append(row)
                test_labels.append(int(digit))
            else:
                train_rows.append(row)
                train_labels.append(int(digit))

    train_x, test_x = np.array(train_rows), np.array(test_rows)   # (2500, 512), (500, 512)
    top = train_x.max()        # biggest number in the TRAINING rows only (the test pile stays unseen)
    train_x, test_x = train_x / top, test_x / top                 # scale to about 0-1, like digits' /16
    return train_x, np.array(train_labels), test_x, np.array(test_labels)