"""Download the human motor-neuron condition (Condition A, Rubin lab) of the Google
in-silico-labeling dataset (Christiansen et al., Cell 2018; CC BY 4.0) for the neural transfer test.

    python scripts/download_isl_neurons.py [--root data/raw/isl_rubin] [--planes 4 5 6 7 8]

Fetches, for every train (22) and test (3) well: the bright-field z-planes requested and the
widefield DAPI (nuclei) max-projection target. Source: gs://in-silico-labeling/paper_data/
{train,test}_single_channel_images/Rubin/ (public HTTPS mirror on storage.googleapis.com).
"""
import argparse
import json
import os
import re
import socket
import sys
import time
import urllib.parse
import urllib.request

socket.setdefaulttimeout(60)  # a stalled connection raises instead of hanging forever

BUCKET = "https://storage.googleapis.com/storage/v1/b/in-silico-labeling/o"
MEDIA = "https://storage.googleapis.com/in-silico-labeling/"


def list_objects(prefix):
    out, token = [], None
    while True:
        q = {"prefix": prefix, "maxResults": 1000, "fields": "items(name,size),nextPageToken"}
        if token:
            q["pageToken"] = token
        for attempt in range(5):
            try:
                d = json.load(urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(q)))
                break
            except Exception:  # noqa: BLE001 - retry listing on network errors
                time.sleep(3)
        else:
            raise RuntimeError("could not list " + prefix)
        out += d.get("items", [])
        token = d.get("nextPageToken")
        if not token:
            return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/isl_rubin")
    ap.add_argument("--planes", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    a = ap.parse_args()
    for split in ("test", "train"):
        objs = list_objects(f"paper_data/{split}_single_channel_images/Rubin/")
        keep = []
        for o in objs:
            n = o["name"]
            if "kind,value-PREDICTED" in n:
                continue
            m = re.search(r"z_depth-(\d+),channel,value-BRIGHTFIELD", n)
            if (m and int(m.group(1)) in a.planes) or "value-DAPI_WIDEFIELD,is_mask-false,kind,value-ORIGINAL" in n:
                keep.append(o)
        os.makedirs(os.path.join(a.root, split), exist_ok=True)
        print(f"{split}: {len(keep)} files, {sum(int(o['size']) for o in keep) / 1e9:.2f} GB")
        for i, o in enumerate(keep):
            well = re.search(r"well-(\w+?),", o["name"]).group(1)
            m = re.search(r"z_depth-(\d+)", o["name"])
            fn = f"{well}_bf_z{int(m.group(1)):02d}.png" if m else f"{well}_dapi.png"
            dst = os.path.join(a.root, split, fn)
            if os.path.exists(dst) and os.path.getsize(dst) == int(o["size"]):
                continue
            for attempt in range(5):
                try:
                    urllib.request.urlretrieve(MEDIA + urllib.parse.quote(o["name"]), dst + ".part")
                    os.replace(dst + ".part", dst)
                    break
                except Exception as e:  # noqa: BLE001 - retry any network error
                    print(f"\n  retry {attempt + 1} for {fn}: {e}")
                    time.sleep(3)
            sys.stdout.write(f"\r  {i + 1}/{len(keep)} {fn}        ")
            sys.stdout.flush()
        print()


if __name__ == "__main__":
    main()
