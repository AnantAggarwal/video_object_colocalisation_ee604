import os
import sys
import shutil
import requests
import argparse
from tqdm import tqdm

DOWNLOAD_LINKS_MAP = {
    'aeroplane': 'https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/aeroplane.tar.gz',
    'bird': 'https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/bird.tar.gz',
    'boat': 'https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/boat.tar.gz',
    'car': 'https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/car.tar.gz',
    'cat': 'https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/cat.tar.gz',
    'cow': 'https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/cow.tar.gz',
    'dog': 'https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/dog.tar.gz',
    'horse': 'https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/horse.tar.gz',
    'motorbike': 'https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/motorbike.tar.gz',
    'train': 'https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/train.tar.gz'
}

def download_and_extract(category, output_dir):
    """
    Downloads and extracts a specific class archive of YouTube-Objects dataset.
    """
    if category not in DOWNLOAD_LINKS_MAP:
        print(f"Error: Category '{category}' is not valid. Choose from: {list(DOWNLOAD_LINKS_MAP.keys())}")
        return False
        
    url = DOWNLOAD_LINKS_MAP[category]
    filename = url.split('/')[-1]
    save_path = os.path.join(output_dir, filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- Downloading YouTube-Objects '{category}' category ---")
    print(f"Source: {url}")
    print(f"Destination: {save_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 8
        
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        progress_bar.close()
        
        print(f"\nDownload completed successfully.")
        print(f"Extracting: {filename} into {output_dir}...")
        
        shutil.unpack_archive(save_path, output_dir)
        print(f"Successfully extracted: {filename}")
        
        os.remove(save_path)
        print(f"Cleaned up archive file.")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Network error downloading {category}: {e}")
    except Exception as e:
        print(f"Error processing {category}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
            
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube-Objects Dataset Downloader")
    parser.add_argument(
        "--category", 
        type=str, 
        default="car", 
        choices=list(DOWNLOAD_LINKS_MAP.keys()),
        help="YouTube-Objects class category to download (default: 'car')"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./data", 
        help="Root directory where the dataset is stored (default: './data')"
    )
    
    args = parser.parse_args()
    success = download_and_extract(args.category, args.output_dir)
    sys.exit(0 if success else 1)
