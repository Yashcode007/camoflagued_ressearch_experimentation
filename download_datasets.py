import os
import sys
import zipfile
import urllib.request
from tqdm import tqdm

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_zip(zip_path, target_dir):
    print(f"Extracting {zip_path} -> {target_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Done extraction.")

def download_isic2016():
    target_dir = os.path.join(DATA_DIR, "ISIC2016")
    os.makedirs(target_dir, exist_ok=True)

    img_url = "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip"
    mask_url = "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip"

    img_zip = os.path.join(target_dir, "test_data.zip")
    mask_zip = os.path.join(target_dir, "test_masks.zip")

    if not os.path.exists(img_zip):
        print("Downloading ISIC 2016 test images...")
        download_url(img_url, img_zip)
    if not os.path.exists(mask_zip):
        print("Downloading ISIC 2016 test masks...")
        download_url(mask_url, mask_zip)

    extract_zip(img_zip, target_dir)
    extract_zip(mask_zip, target_dir)
    print("ISIC 2016 ready at:", target_dir)

def download_kaggle_dataset(slug, target_subdir):
    dest_dir = os.path.join(DATA_DIR, target_subdir)
    os.makedirs(dest_dir, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"Downloading Kaggle dataset '{slug}' into {dest_dir}...")
        api.dataset_download_files(slug, path=dest_dir, unzip=True)
        print(f"Dataset '{slug}' ready at {dest_dir}")
    except Exception as e:
        print(f"Kaggle download failed for {slug}: {e}")
        print("Please ensure kaggle.json is placed in ~/.kaggle/kaggle.json or set KAGGLE_USERNAME and KAGGLE_KEY.")

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "all"

    if action in ["isic", "all"]:
        download_isic2016()
    if action in ["kvasir", "all"]:
        download_kaggle_dataset("debeshjha1/kvasirseg", "Kvasir-SEG")
    if action in ["cod10k", "all"]:
        download_kaggle_dataset("ivanomelchenkoim11/cod10k-dataset", "COD10K")
    if action in ["clinicdb", "all"]:
        download_kaggle_dataset("balraj98/cvcclinicdb", "CVC-ClinicDB")
    if action in ["colondb", "all"]:
        download_kaggle_dataset("longvil/cvc-colondb", "CVC-ColonDB")
    if action in ["etis", "all"]:
        download_kaggle_dataset("nguyenvoquocduong/etis-laribpolypdb", "ETIS-LaribPolypDB")
