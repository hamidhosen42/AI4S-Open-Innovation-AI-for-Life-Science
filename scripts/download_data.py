"""Download the HeLa "Kyoto" paired bright-field / fluorescence training set.

Source: Zenodo 10.5281/zenodo.6140064 (CC BY 4.0), derived from 10.5281/zenodo.6139958.
Usage: python scripts/download_data.py [--root data/raw]
The archive is verified (size + md5) and the extraction is checked (1500 train / 750 test files).
"""
import argparse
import glob
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile

URL = "https://zenodo.org/records/6140064/files/training_dataset.zip?download=1"
MD5 = "7d218466d217fd62dc8ec56ad76d23d7"
SIZE = 756_628_102


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def complete(out):
    return len(glob.glob(os.path.join(out, "train", "*.tif"))) == 1500 and len(glob.glob(os.path.join(out, "test", "*.tif"))) == 750


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    a = ap.parse_args()
    os.makedirs(a.root, exist_ok=True)
    zpath = os.path.join(a.root, "training_dataset.zip")
    out = os.path.join(a.root, "hela_kyoto")
    if complete(out):
        print("already present and complete:", out)
        return
    if not (os.path.exists(zpath) and os.path.getsize(zpath) == SIZE and md5(zpath) == MD5):
        print("downloading ~757 MB ...")

        def hook(b, bs, total):
            sys.stdout.write(f"\r{b * bs / max(total, 1) * 100:5.1f}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(URL, zpath + ".part", hook)
        print()
        if os.path.getsize(zpath + ".part") != SIZE or md5(zpath + ".part") != MD5:
            raise SystemExit("download incomplete or corrupted (size/md5 mismatch) - please re-run")
        os.replace(zpath + ".part", zpath)
    print("extracting ...")
    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(tmp)
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    if not complete(out):
        raise SystemExit("extraction incomplete - expected 1500 train and 750 test files")
    print("done:", out)


if __name__ == "__main__":
    main()
