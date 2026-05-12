"""Prepare extracted HAM10000 metadata/images into train/test folders."""

from __future__ import annotations

import argparse

from data import prepare_ham10000


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HAM10000 folder dataset")
    parser.add_argument("--raw_dir", default="data/ham10000_raw")
    parser.add_argument("--output_dir", default="data/ham10000")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prepare_ham10000(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        resize=args.resize,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
