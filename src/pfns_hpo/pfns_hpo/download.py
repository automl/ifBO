import argparse
from pathlib import Path
import requests
import shutil
import zipfile
import tarfile
import os


VERSION_MAP = {
    "0.0.1": dict(
        url="https://ml.informatik.uni-freiburg.de/research-artifacts/ifbo/surrogate.tar.gz",
        name="surrogate",
        final_name="bopfn_broken_unisep_1000curves_10params_2M",
        extension="pt"
    )
}

# Helper functions to generate the file names
FILENAME = lambda version: f"{VERSION_MAP.get(version).get('name')}.tar.gz"
WEIGHTS_FILE_NAME = lambda version: (
    f"{VERSION_MAP.get(version).get('name')}.{VERSION_MAP.get(version).get('extension')}"
)
WEIGHTS_FINAL_NAME = lambda version: (
    f"{VERSION_MAP.get(version).get('final_name')}.{VERSION_MAP.get(version).get('extension')}"
)


def download_and_decompress(url: str, path: Path, version: str="0.0.1") -> bool:
    """ Helper function to download a file from a URL and decompress it and store by given name.
    """
    # Send a HTTP request to the URL of the file
    response = requests.get(url, allow_redirects=True)

    success_flag = True
    # Check if the request is successful
    if response.status_code == 200:
        # Save the .tar.gz file
        with open(path, 'wb') as f:
            f.write(response.content)

        # Decompress the .tar.gz file
        if path.name.endswith('.tar.gz') and path.exists:
            os.system(f"tar -xvf {path} -C {path.parent.absolute()}")
        else:
            success_flag = False
            print(f"Failed to find surrogate file at {path}!")

        # Rename the decompressed file
        _path = path.parent.absolute() / WEIGHTS_FILE_NAME(version)
        _final_path = path.parent.absolute() / WEIGHTS_FINAL_NAME(version)
        if _path.exists():
            os.rename(_path, path.parent.absolute() / _final_path)
        else:
            success_flag = False
            print(f"Failed to find the decompressed file at {_path}!")

        # Check for the final file
        if not _final_path.exists():
            success_flag = False
    else:
        success_flag = False

    return success_flag


def parse_args() -> argparse.Namespace:
    """ Helper function to parse the command line arguments.
    """
    args = argparse.ArgumentParser()

    args.add_argument(
        "--version",
        type=str,
        default="0.0.1",
        help="The version of the PFN model to download"
    )
    args.add_argument(
        "--path",
        type=str,
        default=None,
        help="The path to save the downloaded file"
    )

    parser = args.parse_args()
    return parser


if __name__ == "__main__":

    args = parse_args()

    assert args.version in VERSION_MAP, "The version provided is not available"

    if args.path is None:
        args.path = Path(__file__).parent.absolute() / ".." / ".." / "PFNS4HPO" / "final_models"
    else:
        args.path = Path(args.path)

    if not args.path.exists():
        os.makedirs(args.path)

    # Use the function
    if download_and_decompress(
        url=VERSION_MAP.get(args.version).get("url"),
        path=args.path / FILENAME(args.version),
        version=args.version
    ):
        print(f"Successfully downloaded FT-PFN v{args.version} in to {args.path}!")
    else:
        print(f"Failed to download FT-PFN v{args.version} in to {args.path}!")
