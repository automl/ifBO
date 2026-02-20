import argparse
import os
from pathlib import Path
import tarfile
import time
import random
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
SURROGATE_URL: str = (
    _url
    if (_url := os.getenv("IFBO_SURROGATE_URL")) and _is_valid_url(_url)
    else DEFAULT_URL
)
assert _is_valid_url(SURROGATE_URL), f"The surrogate URL provided is not valid: {SURROGATE_URL}"

VERSION_MAP: dict[str, dict[str, str]] = {
    "0.0.1": {
        "url": SURROGATE_URL,
        "name": "ftpfnv0.0.1",
        "final_name": "bopfn_broken_unisep_1000curves_10params_2M",
        "extension": "pt",
    }
}

# Environment variable configuration for download retries and timeouts
MAX_RETRIES: int = int(os.getenv("IFBO_MAX_RETRIES", "5"))  # NOTE: exponential increase with retries
WAIT_TIME: int = int(os.getenv("IFBO_WAIT_TIME", "5"))  # in seconds
REQUEST_TIMEOUT: int = int(os.getenv("IFBO_REQUEST_TIMEOUT", "60"))  # in seconds


# Helper functions to generate the file names
def FILENAME(version: str) -> str:
    return f"{VERSION_MAP[version].get('name')}.tar.gz"


def FILE_URL(version: str) -> str:
    return f"{VERSION_MAP[version].get('url')}"


def WEIGHTS_FILE_NAME(version: str) -> str:
    return f"{VERSION_MAP[version].get('name')}.{VERSION_MAP[version].get('extension')}"


def WEIGHTS_FINAL_NAME(version: str) -> str:
    return f"{VERSION_MAP[version].get('final_name')}.{VERSION_MAP[version].get('extension')}"


def _fetch_response_with_repeats(url: str, max_retries: int | None = None) -> requests.Response:
    max_retries = max_retries or MAX_RETRIES
    
    for attempt in range(max_retries):
        response = requests.get(url, allow_redirects=True, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            print(f"Successfully fetched the surrogate from {url} on attempt {attempt + 1}.")
            return response

        elif response.status_code in [202, 403, 429, 502, 503, 504]:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                base_wait = WAIT_TIME * (2 ** attempt)
                jitter = 0.1 * base_wait * random.uniform(0, 1)  # to stagger parallel retries 
                wait_time = base_wait + jitter
                print(
                    f"Received HTTP {response.status_code}. "
                    f"Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
            else:
                raise ValueError(
                    f"Failed to download the surrogate from {url} after {max_retries} attempts. "
                    f"Received HTTP status code: {response.status_code}. "
                    "Please either try again later, use an alternative link or contact the authors through github."
                )
        else:
            raise ValueError(
                f"Failed to download the surrogate from {url}. "
                f"Received HTTP status code: {response.status_code}. "
                "Please either try again later, use an alternative link or contact the authors through github."
            )
    raise ValueError(f"Failed to download after {max_retries} attempts")


def download_and_decompress(url: str, path: Path, max_retries: int | None  = None) -> None:
    """Helper function to download a file from a URL and decompress it and store by given name.

    Args:
        url (str): URL of the file to download
        path (Path): Path along with filename to save the downloaded file
        max_retries (int | None): Maximum number of retry attempts

    Returns:
        None
    """
    # Check if the file already exists
    path = Path(path) if not isinstance(path, Path) else path
    if path.exists():
        return

    # Send a HTTP request to the URL of the file
    response = _fetch_response_with_repeats(url, max_retries)

    # Save the .tar.gz file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

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
