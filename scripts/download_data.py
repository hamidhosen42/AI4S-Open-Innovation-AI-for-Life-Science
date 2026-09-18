"""Download the HeLa "Kyoto" paired bright-field / fluorescence training set.

Source: Zenodo 10.5281/zenodo.6140064 (CC BY 4.0), derived from 10.5281/zenodo.6139958.
Usage: python scripts/download_data.py [--root data/raw]
"""
import argparse
import os
import sys
import urllib.request
import zipfile

URL = "https://zenodo.org/records/6140064/files/training_dataset.zip?download=1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    a = ap.parse_args()
    os.makedirs(a.root, exist_ok=True)
    zpath = os.path.join(a.root, "training_dataset.zip")
    out = os.path.join(a.root, "hela_kyoto")
    if os.path.isdir(os.path.join(out, "train")):
        print("already present:", out)
        return
    if not os.path.exists(zpath):
        print("downloading ~757 MB ...")

        def hook(b, bs, total):
            done = b * bs / max(total, 1)
            sys.stdout.write(f"\r{done*100:5.1f}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(URL, zpath, hook)
        print()
    print("extracting ...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    print("done:", out)


if __name__ == "__main__":
    main()
