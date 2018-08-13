# roboarchive-broom
A toolbox to clean archive documents

## Install dependencies

1. Python 3.6
2. Keras
3. OpenCV

## Unpack Samples

Unpack samples into the project dir, it should be two directories in it:

* "samples-raw" (x-train data), the tiles are of 64x64 size. The central 32x32 area does not
intersect with neighbour tiles (and 16 + 16 pixels padding does).
* "samples-clean" (y-train data). The tiles are of 32x32 size and they don't include any padding.

## Run the script

```bash
python3.6 train_cnn.py
```
