import gdown
import os

def download_from_gdrive(file_url, output_path="."):
    """
    Download a file from Google Drive using gdown.

    :param file_url: Shared Google Drive link (string)
    :param output_path: Save path or file name (default: current dir)
    """
    # Make sure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    print("Downloading...")
    gdown.download(url=file_url, output=output_path, quiet=False, fuzzy=True)
    print("Downloaded to:", output_path)


if __name__ == "__main__":
    # Example Google Drive share link
    gdrive_url = "https://drive.google.com/file/d/1er5NOTuWgO3fUeSvmWzUSzubtzz4vfBC/view?usp=sharing"

    # Output location (change this)
    save_as = "../../../../../../birul001/BIRUL001/data/wikiart/wikiart.zip"   # could also be "my_model.zip"

    download_from_gdrive(gdrive_url, save_as)
