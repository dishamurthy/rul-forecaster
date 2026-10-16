"""
Download NASA C-MAPSS FD001 into data/raw/.

    python scripts/download_data.py

Sources tried in order:
  1. NASA's official archive  (CMAPSSData.zip -> extracts the 3 FD001 files)
  2. Hugging Face dataset mirror (individual .txt files)

If both fail (offline / firewall), download CMAPSSData.zip yourself from
https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data  (or Kaggle: "NASA CMAPSS"),
unzip it, and copy train_FD001.txt, test_FD001.txt and RUL_FD001.txt into data/raw/.
"""
import io
import shutil
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "data" / "raw"
FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]

NASA_ZIP = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
HF_BASE = "https://huggingface.co/datasets/SoyVitou/NASA-C-MAPSS-Turbofan-Engine/resolve/main/data/"


def _ctx():
    """macOS python.org builds often lack root certs; use certifi if it is installed."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return None


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (rul-forecaster)"})
    with urllib.request.urlopen(req, timeout=120, context=_ctx()) as r:
        return r.read()


def have_all() -> bool:
    return all((DEST / f).exists() and (DEST / f).stat().st_size > 0 for f in FILES)


def from_nasa_zip() -> bool:
    print(f"[1/2] NASA archive: {NASA_ZIP}")
    data = _fetch(NASA_ZIP)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = {Path(n).name: n for n in z.namelist()}
        for f in FILES:
            if f not in names:
                raise FileNotFoundError(f"{f} not found inside the zip")
            with z.open(names[f]) as src, open(DEST / f, "wb") as out:
                shutil.copyfileobj(src, out)
    return True


def from_huggingface() -> bool:
    print(f"[2/2] Hugging Face mirror: {HF_BASE}")
    for f in FILES:
        (DEST / f).write_bytes(_fetch(HF_BASE + f))
    return True


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    if have_all():
        print(f"already present in {DEST}")
        return 0
    for step in (from_nasa_zip, from_huggingface):
        try:
            step()
            if have_all():
                print(f"done -> {DEST}")
                return 0
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {type(e).__name__}: {e}")
    print("\nAll automatic downloads failed. Manual option:\n"
          "  1. Download CMAPSSData.zip from https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data\n"
          "     (or Kaggle: search 'NASA CMAPSS Jet Engine Simulated Data')\n"
          "  2. Unzip it and copy train_FD001.txt, test_FD001.txt, RUL_FD001.txt into:\n"
          f"     {DEST}")
    return 1


if __name__ == "__main__":
    sys.exit(main())