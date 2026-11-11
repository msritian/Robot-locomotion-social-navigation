"""Pull only the EVT-Bench MP3D scenes out of mp3d_habitat.zip (16 GB) using HTTP range requests.

Running this confirms you have agreed to the Matterport3D terms of use
(http://kaldir.vc.cit.tum.de/matterport/MP_TOS.pdf).

    python3 evt/fetch_mp3d_scenes.py -o data/scene_datasets          # the 14 EVT-Bench scenes
    python3 evt/fetch_mp3d_scenes.py --list                          # just print what would be fetched
"""
import argparse
import os
import shutil
import urllib.request
import zipfile

URL = "https://kaldir.vc.cit.tum.de/matterport/v1/tasks/mp3d_habitat.zip"
EVT_SCENES = [
    "E9uDoFAP3SH", "2n8kARJN3HM", "ac26ZMwG7aT", "gYvKGZ5eRqb", "VLzqgDo317F", "Vt2qJdWjCF2", "V2XKFyX4ASd",
    "pRbA3pwrgk9", "oLBMNvg9in8", "YFuZgdQ5vWj", "e9zR4mvMWw7", "XcA2TqTSSAj", "JmbYfDe2QKZ", "HxpKQynjfin",
]
BLOCK = 8 << 20


class HttpFile:
    """Read-only seekable file over HTTP range requests, with one cached block."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])
        self.buf_start, self.buf = -1, b""

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _fetch(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req) as r:
            return r.read()

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            if not (self.buf_start <= self.pos < self.buf_start + len(self.buf)):
                self.buf_start = self.pos
                self.buf = self._fetch(self.pos, min(self.size, self.pos + max(BLOCK, n)))
            i = self.pos - self.buf_start
            chunk = self.buf[i:i + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out_dir", default="data/scene_datasets")
    ap.add_argument("--scenes", nargs="+", default=EVT_SCENES)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    zf = zipfile.ZipFile(HttpFile(URL))
    want = [i for i in zf.infolist() if not i.is_dir() and any(f"/{s}/" in "/" + i.filename for s in args.scenes)]
    found = {s for s in args.scenes if any(f"/{s}/" in "/" + i.filename for i in want)}
    print(f"{len(want)} files, {sum(i.file_size for i in want) / 1e9:.2f} GB, scenes found {len(found)}/{len(args.scenes)}")
    for s in sorted(set(args.scenes) - found):
        print("  missing:", s)
    if args.list:
        for i in want:
            print(f"  {i.file_size / 1e6:8.1f} MB  {i.filename}")
        return
    for k, i in enumerate(want, 1):
        dst = os.path.join(args.out_dir, i.filename)
        if os.path.exists(dst) and os.path.getsize(dst) == i.file_size:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with zf.open(i) as src, open(dst + ".part", "wb") as f:
            shutil.copyfileobj(src, f, BLOCK)
        os.replace(dst + ".part", dst)
        print(f"[{k}/{len(want)}] {i.filename}")


if __name__ == "__main__":
    main()
