"""
Decode the FLAME analysis output file (results.tar.gz.b64) and extract all files.

Usage:
    python decode_results.py results.tar.gz.b64
    python decode_results.py results.tar.gz.b64 --output my_folder

    python3 decode_results.py results.tar.gz.b64-8.txt
"""

import argparse
import base64
import io
import os
import sys
import tarfile


def decode_results(input_file: str, output_dir: str = "output") -> None:
    print(f"Reading: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        b64_content = f.read().strip()

    print(f"Decoding base64 ({len(b64_content):,} chars)...")
    tar_bytes = base64.b64decode(b64_content)
    print(f"Decoded to {len(tar_bytes):,} bytes")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Extracting to '{output_dir}/'...")
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        members = tar.getmembers()
        print(f"  {len(members)} files in archive:")
        for member in members:
            size = f"{member.size:,} bytes" if member.isfile() else "dir"
            print(f"    {member.name}  ({size})")
        tar.extractall(output_dir)

    print(f"\nDone — files extracted to '{output_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode FLAME analysis results")
    parser.add_argument("input", help="Path to results.tar.gz.b64 file")
    parser.add_argument(
        "--output", "-o", default="output",
        help="Output directory (default: output/)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: file not found: {args.input}")
        sys.exit(1)

    decode_results(args.input, args.output)
