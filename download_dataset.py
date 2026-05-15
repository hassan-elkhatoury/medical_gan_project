"""Download and extract the LC25000 lung/colon histopathology dataset from Kaggle."""

from __future__ import annotations

import argparse
import os
import subprocess
import zipfile
from pathlib import Path


def download_lung_cancer_dataset(
    output_dir: str = "data/lung_cancer_raw",
    dataset: str = "andrewmvd/lung-and-colon-cancer-histopathological-images",
    force: bool = False,
) -> Path:
    """Download the Kaggle dataset archive and extract it."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("KAGGLE_USERNAME") and not (Path.home() / ".kaggle" / "kaggle.json").exists():
        raise RuntimeError(
            "Kaggle credentials were not found. Add ~/.kaggle/kaggle.json or set "
            "KAGGLE_USERNAME and KAGGLE_KEY before running this script."
        )

    zip_path = output_path / "lung-and-colon-cancer-histopathological-images.zip"
    command = [
        "kaggle",
        "datasets",
        "download",
        dataset,
        "-p",
        str(output_path),
    ]
    if force:
        command.append("--force")
    subprocess.run(command, check=True)

    if not zip_path.exists():
        matches = list(output_path.glob("*.zip"))
        if not matches:
            raise FileNotFoundError(f"No downloaded zip file found in {output_path}")
        zip_path = matches[0]

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_path)
    print(f"Extracted lung cancer histopathology dataset to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LC25000 from Kaggle")
    parser.add_argument("--output_dir", default="data/lung_cancer_raw")
    parser.add_argument(
        "--dataset",
        default="andrewmvd/lung-and-colon-cancer-histopathological-images",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_lung_cancer_dataset(args.output_dir, args.dataset, args.force)


if __name__ == "__main__":
    main()
