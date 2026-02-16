import argparse
import os
from pathlib import Path
import tarfile
from urllib.parse import urlparse

import requests


def _is_valid_url(url: str) -> bool:
    """Check if string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

# From https://figshare.com/articles/dataset/IfBO_surrogate-compressed/31286173?file=61709839
DEFAULT_URL = "https://api.figshare.com/v2/file/download/61709839"
# The URL can be custom adjusted through the environment variable `IFBO_SURROGATE_URL`
_url = None
SURROGATE_URL = (
    _url
    if (_url := os.getenv("IFBO_SURROGATE_URL")) and _is_valid_url(_url)
    else DEFAULT_URL
)


VERSION_MAP = {
    "0.0.1": dict{
        "url": SURROGATE_URL,
        "name": "ftpfnv0.0.1",
        "final_name": "bopfn_broken_unisep_1000curves_10params_2M",
        "extension": "pt",
    }
}


# Helper functions to generate the file names
def FILENAME(version: str) -> str:
    return f"{VERSION_MAP[version].get('name')}.tar.gz"


def FILE_URL(version: str) -> str:
    return f"{VERSION_MAP[version].get('url')}"


def WEIGHTS_FILE_NAME(version: str) -> str:
    return f"{VERSION_MAP[version].get('name')}.{VERSION_MAP[version].get('extension')}"


def WEIGHTS_FINAL_NAME(version: str) -> str:
    return f"{VERSION_MAP[version].get('final_name')}.{VERSION_MAP[version].get('extension')}"


def download_and_decompress(url: str, path: Path) -> None:
    """Helper function to download a file from a URL and decompress it and store by given name.

    Args:
        url (str): URL of the file to download
        path (Path): Path along with filename to save the downloaded file

    Returns:
        bool: Flag to indicate if the download and decompression was successful
    """
    # Check if the file already exists
    if path.exists():
        return

    # Send a HTTP request to the URL of the file
    response = requests.get(url, allow_redirects=True)

    # Check if the request is successful
    if response.status_code != 200:
        raise ValueError(
            f"Failed to download the surrogate from {url}."
            f" Recieved HTTP status code: {response.status_code}."
            "Please either try again later, use an alternative link or contact the authors through github."
        )

    # Save the .tar.gz file
    with open(path, "wb") as f:
        f.write(response.content)

    # Decompress the .tar.gz file
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(path.parent.absolute())


def parse_args() -> argparse.Namespace:
    """Helper function to parse the command line arguments."""
    args = argparse.ArgumentParser()

    args.add_argument(
        "--version", type=str, default="0.0.1", help="The version of the PFN model to download"
    )
    args.add_argument(
        "--path", type=str, default=None, help="The path to save the downloaded file"
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
        url=VERSION_MAP[args.version]["url"], path=args.path / FILENAME(args.version)
    )
    print(f"Successfully downloaded FT-PFN v{args.version} in to {args.path}!")
