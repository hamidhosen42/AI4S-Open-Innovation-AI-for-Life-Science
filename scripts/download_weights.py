"""Download the released ChipStain checkpoint into weights/chipstain.pt.

The weights are published as a GitHub release asset of this repository.
"""
import argparse
import os
import sys
import urllib.request

URL = "https://github.com/hamidhosen42/AI4S-Open-Innovation-AI-for-Life-Science/releases/download/v0.1.0/chipstain.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default="weights/chipstain.pt")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if os.path.exists(a.out):
        print("already present:", a.out)
        return
    print("downloading", a.url)

    def hook(b, bs, total):
        sys.stdout.write(f"\r{b * bs / max(total, 1) * 100:5.1f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(a.url, a.out, hook)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
