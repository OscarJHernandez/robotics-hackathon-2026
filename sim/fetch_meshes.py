"""Download the JetHexa meshes from Hiwonder's GitHub repo into sim/meshes/ and check their hashes.

    python sim/fetch_meshes.py

The meshes are not stored in this repo: Hiwonder publishes them for educational and research use but
without a license. The files are pinned to one commit, and each download is checked against its SHA-256.
"""

import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

MESH_DIR = Path(__file__).resolve().parent / "meshes"
COMMIT = "6968b1ab141c51a7dc959a1ff232a469de8db3dd"
URL = "https://raw.githubusercontent.com/Hiwonder/JetHexa/{}/jethexa_description/meshes/simplify/{}"
MESHES = {
    "L_femur.stl": "37a19d10c5e1fc1845c12f8f901043ccf802a9bf39ec34ef3c6352adc019ccf8",
    "L_tibia_a.stl": "dab744403709ea5209cd63b2dc72a05e0bb6a4bcd7180a41b1e0925fb961ee38",
    "L_tibia_b.stl": "e5f1f93a4e04ccae64acfacf239b9d4acb37215d734c9ff4163875fcb8c52602",
    "R_femur.stl": "8cf1731e403d6432970c82197a3fa9d0a5a2f365fa3dddc07df20c484bcb24c7",
    "R_tibia_a.stl": "20ae54ebe95069133f1f11b1d3a23d9798fc7edeaf97b8cd4d98f6f94f562d3b",
    "R_tibia_b.stl": "e00c2e54f44eaf01bb87ad3b64751a56a72d4f990b8faf29d4bf107b33c497ec",
    "body_a.stl": "5c285239f8bd5443aba3af3ca1d44b622013222a52cadebe5bdeb2ca39d46354",
    "body_b.stl": "f87985ce40352ace6e295a733e26d1ef0410bd9e4dc8a732b62bbe540d933a46",
    "coxa.stl": "ccb6432277901d71ed729fbd7ee1f5c5db8cf6a82f1b061a3d0f3709faaa964e",
    "dabai.stl": "05beba58d6594b0103832bb58005a81e2df329b2a66d1a7aeb3e36e4abbe6ae3",
    "dabai_len.stl": "facb9cde95604763459bb241abb7b7d3f0db9509890788d6edc70389742a5d1a",
    "g4.stl": "50acd24ee87115b474d00505b6fa873f6cefb8625b434d0c6ef6efe4cefc861c",
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meshes_available():
    return all((MESH_DIR / name).is_file() for name in MESHES)


def fetch():
    MESH_DIR.mkdir(exist_ok=True)
    for name, expected in sorted(MESHES.items()):
        target = MESH_DIR / name
        if target.is_file() and _sha256(target) == expected:
            print("ok      ", name)
            continue
        partial = target.with_suffix(".partial")
        with urllib.request.urlopen(URL.format(COMMIT, name), timeout=60) as response:
            partial.write_bytes(response.read())
        actual = _sha256(partial)
        if actual != expected:
            partial.unlink()
            raise RuntimeError("{}: hash {} does not match the pinned {}".format(name, actual, expected))
        partial.replace(target)
        print("fetched ", name)


def main():
    try:
        fetch()
    except (OSError, RuntimeError, urllib.error.URLError) as error:
        print("mesh download failed: {}".format(error), file=sys.stderr)
        return 1
    print("meshes are in", MESH_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
