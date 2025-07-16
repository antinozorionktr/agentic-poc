# download_coco.py
from pathlib import Path
import yaml
from ultralytics.utils.downloads import download

# Load the YAML file
with open("coco.yaml", "r") as f:
    data = yaml.safe_load(f)

# Get dataset root path
dir = Path(data["path"])
segments = True  # segment or box labels

# Download labels
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
urls = [url + ("coco2017labels-segments.zip" if segments else "coco2017labels.zip")]
download(urls, dir=dir.parent)

# Download images
urls = [
    "http://images.cocodataset.org/zips/train2017.zip",  # 19 GB
    "http://images.cocodataset.org/zips/val2017.zip",    # 1 GB
    "http://images.cocodataset.org/zips/test2017.zip",   # 7 GB (optional)
]
download(urls, dir=dir / "images", threads=3)
