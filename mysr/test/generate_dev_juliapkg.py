"""Switch MySR's JuliaPkg configuration to a local MySRCore checkout."""

import argparse
import json
from pathlib import Path


def generate_dev_config(juliapkg_json: Path, path_to_mysrcore: Path) -> None:
    with juliapkg_json.open("r", encoding="utf-8") as file:
        juliapkg = json.load(file)

    package = juliapkg["packages"]["MySRCore"]
    package.pop("url", None)
    package.pop("rev", None)
    package["path"] = str(path_to_mysrcore)
    package["dev"] = True

    with juliapkg_json.open("w", encoding="utf-8") as file:
        json.dump(juliapkg, file, indent=4)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use a local MySRCore.jl checkout instead of the pinned release."
    )
    parser.add_argument("juliapkg_json", type=Path)
    parser.add_argument("path_to_mysrcore", type=Path)
    args = parser.parse_args()

    generate_dev_config(args.juliapkg_json, args.path_to_mysrcore)


if __name__ == "__main__":
    main()
