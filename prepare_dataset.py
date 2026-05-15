"""Prepare extracted COVID-19 radiography images into train/test folders."""

from __future__ import annotations

import argparse

from data import prepare_covid_radiography


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare COVID-19 radiography folder dataset")
    parser.add_argument("--raw_dir", default="data/covid_radiography_raw")
    parser.add_argument("--output_dir", default="data/covid_radiography")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_images_per_class", type=int, default=None)
    args = parser.parse_args()

    prepare_covid_radiography(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        resize=args.resize,
        overwrite=args.overwrite,
        max_images_per_class=args.max_images_per_class,
    )


if __name__ == "__main__":
    main()
