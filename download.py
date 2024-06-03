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
        name="bopfn_broken_unisep_1000curves_10params_2M",
    )
}


def download_and_decompress(url: str, path: Path) -> None:
    """ Helper function to download a file from a URL and decompress it and store by given name.
    """

    if isinstance(path, Path):
        path = str(path)

    # Send a HTTP request to the URL of the file
    response = requests.get(url, stream=True)

    # Check if the request is successful
    if response.status_code == 200:
        # Write the contents of the response to a file
        with open(path, 'wb') as file:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, file)
        
        print(f"Downloaded the file to {path}")

        # If it's a zip file
        if path.endswith('.zip'):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(path))
            print(f"Decompressed the zip file at {path}")

        # If it's a tar.gz file
        elif path.endswith('.tar.gz'):
            with tarfile.open(path, 'r:gz') as tar_ref:
                tar_ref.extractall(os.path.dirname(path))
            print(f"Decompressed the tar.gz file at {path}")

    else:
        print(f"Failed to download the file from {url}")


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
    download_and_decompress(
        VERSION_MAP.get(args.version).get("url"), 
        args.path / f"{VERSION_MAP.get(args.version).get('name')}.pt"
    )
    print(f"Successfully downloaded FT-PFN v{args.version} in to {args.path}!")

