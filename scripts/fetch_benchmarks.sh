#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
out="$root/examples/data/benchmarks"
mkdir -p "$out"

download() {
  name=$1; url=$2; sha=$3
  dest="$out/$name"
  curl --fail --location --retry 3 --silent --show-error "$url" -o "$dest"
  actual=$(shasum -a 256 "$dest" | awk '{print $1}')
  [ "$actual" = "$sha" ] || { echo "checksum mismatch for $name: $actual" >&2; exit 1; }
}

# Public RCSB coordinate files and the public Xylanase tutorial curve.
download 1ubq.pdb https://files.rcsb.org/download/1UBQ.pdb d4a6812d8951cf6594e6a0763f089e35f5a80b62acb3c117b2c5565228a7b161
download 2dfc.pdb https://files.rcsb.org/download/2DFC.pdb 3fff1019f69b697f4b4c0260f0b2324e9cda25279fadc83253f0ea2a191a066e
download 1aon.pdb https://files.rcsb.org/download/1AON.pdb 5cddcf0ebe222238073f73f4fcc0f579a090389f9c3d152399b4e41ca4a0ed3e
download xylanase.dat https://sastutorials.org/Proteins/part1/X4_Xyl-SS-Gt.dat 8d66c1920613ede751140a3c70995bbcdccdcfc746acb2a60a3c9bf9d5dacb5a

echo "Benchmark files are ready in $out"
