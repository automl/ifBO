from pathlib import Path
import requests
import os

VERSION_MAP = {
    "0.0.1": dict(
        url="https://ml.informatik.uni-freiburg.de/research-artifacts/ifbo/ftpfnv0.0.1.tar.gz",
        name="ftpfnv0.0.1",
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
            os.system(f"tar -xvf {path} -C {path.parent.absolute()} > /dev/null 2>&1")
        else:
            success_flag = False
            print(f"Failed to find surrogate file at {path}!")
    else:
        success_flag = False

    if success_flag:
        print(f"Successfully downloaded and decompressed the file at {path}!")
    else:
        print(f"Failed to download and decompress the file at {path}!")
    
    return success_flag
