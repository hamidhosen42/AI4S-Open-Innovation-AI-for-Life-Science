"""Download the released ChipStain checkpoint (seed-0 ChipStain, beta-NLL) into weights/chipstain.pt.

The weights are published as a GitHub release asset of this repository and verified by sha256.
    python scripts/download_weights.py [--force]
"""
import argparse
import hashlib
import os
import sys
import urllib.request

URL = "https://github.com/hamidhosen42/AI4S-Open-Innovation-AI-for-Life-Science/releases/download/v0.1.0/chipstain.pt"
SHA256 = "d72951a79279bb7db02cd3941a7e5fa8d13819fe513dc0249620aa3513937bcb"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default="weights/chipstain.pt")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if os.path.exists(a.out) and not a.force:
        if sha256(a.out) == SHA256:
            print("already present and verified:", a.out)
            return
        print("existing file has a different checksum - downloading again")
    print("downloading", a.url)

    def hook(b, bs, total):
        sys.stdout.write(f"\r{b * bs / max(total, 1) * 100:5.1f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(a.url, a.out + ".part", hook)
    print()
    if sha256(a.out + ".part") != SHA256:
        raise SystemExit("checksum mismatch - the download is incomplete or the release file differs")
    os.replace(a.out + ".part", a.out)
    print("saved and verified", a.out)


if __name__ == "__main__":
    main()
