"""Multi-threaded chunked downloader for the Paderborn archives.

A single connection is throttled to ~11 KB/s; eight concurrent Range requests
reach ~1 MB/s, so the file is pulled in parallel chunks.
Usage: python -m src.fetch_pu K001 KA04 KI04
"""
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"
OUT = os.path.join("data", "pu")
N_THREADS = 12
CHUNK = 4 * 1024 * 1024


def remote_size(url):
    r = requests.head(url, timeout=30, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def fetch_range(url, start, end, path, idx):
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(4):
        try:
            with requests.get(url, headers=headers, timeout=120, stream=True) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for blk in r.iter_content(1 << 16):
                        f.write(blk)
            return
        except Exception as e:  # retry
            if attempt == 3:
                raise
            print(f"  retry {idx}: {e}")


def download(name):
    url = BASE + name + ".rar"
    final = os.path.join(OUT, name + ".rar")
    if os.path.exists(final) and os.path.getsize(final) > 0:
        print(f"{name}: already present, skipped")
        return
    size = remote_size(url)
    parts = max(1, min(N_THREADS * 2, size // CHUNK + 1))
    bounds = [(i * size // parts, (i + 1) * size // parts - 1) for i in range(parts)]
    tmp = [final + f".part{i}" for i in range(parts)]
    print(f"{name}: {size/1e6:.0f} MB in {parts} chunks")
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futs = [ex.submit(fetch_range, url, s, e, tmp[i], i)
                for i, (s, e) in enumerate(bounds)]
        for f in futs:
            f.result()
    with open(final, "wb") as out:
        for t in tmp:
            with open(t, "rb") as f:
                out.write(f.read())
            os.remove(t)
    print(f"{name}: done, {os.path.getsize(final)/1e6:.0f} MB")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or ["K001", "K002", "K003", "KA04", "KA15", "KA16",
                             "KI04", "KI14", "KI16"]
    for n in names:
        download(n)
