"""Download and extract the COVID-19 Radiography Database from Kaggle."""

from __future__ import annotations

import argparse
import os
import subprocess
import zipfile
from pathlib import Path


def download_covid_radiography_dataset(
    output_dir: str = "data/covid_radiography_raw",
    dataset: str = "tawsifurrahman/covid19-radiography-database",
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

    zip_path = output_path / "covid19-radiography-database.zip"
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
    print(f"Extracted COVID-19 Radiography Database to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download COVID-19 Radiography Database from Kaggle")
    parser.add_argument("--output_dir", default="data/covid_radiography_raw")
    parser.add_argument(
        "--dataset",
        default="tawsifurrahman/covid19-radiography-database",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_covid_radiography_dataset(args.output_dir, args.dataset, args.force)


if __name__ == "__main__":
    main()
