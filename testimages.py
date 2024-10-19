import os
import requests
from PIL import Image
from duckduckgo_search import DDGS
from fastcore.all import *
from urllib.parse import urlparse
import wget  # Install using pip install wget


def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    ddgs = DDGS()
    return [result['image'] for result in ddgs.images(keywords=term, max_results=max_images)]

def test_search_images():
    term = "cat"
    max_images = 5
    images = search_images(term, max_images)
    print(f"Found {len(images)} images for '{term}':")
    for image in images:
        print(image)

# Call the test function
test_search_images()

def download_image(url, folder):
    try:
        # Ensure target folder exists
        os.makedirs(folder, exist_ok=True)

        # Extract file name from URL and sanitize it
        filename = os.path.basename(urlparse(url).path)
        filepath = os.path.join(folder, filename)

        # Download image using requests
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, allow_redirects=True)
        response.raise_for_status()

        # Save image to file
        with open(filepath, 'wb') as f:
            f.write(response.content)

        return filepath
    except requests.exceptions.HTTPError as e:
        if response.status_code == 403:
            print(f"Access forbidden for image {url}: {e}")
        else:
            print(f"HTTP error occurred while downloading image {url}: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image {url}: {e}")
    except Exception as e:
        print(f"Other error downloading image {url}: {e}")
    return None

def resize_images(image_paths, output_folder, target_size=(1280, 720)):
    resized_paths = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Image file {image_path} does not exist. Skipping.")
            continue
        try:
            image = Image.open(image_path)
            image = image.resize(target_size)
            resized_path = os.path.join(output_folder, os.path.basename(image_path))
            image.save(resized_path)
            resized_paths.append(resized_path)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    return resized_paths

def test_download_image():
    url = "https://www.rd.com/wp-content/uploads/2021/01/GettyImages-1175550351.jpg?w=2141"
    folder = "./downloads"
    downloaded_image = download_image(url, folder)
    if downloaded_image:
        print(f"Image downloaded successfully: {downloaded_image}")
    else:
        print(f"Failed to download image from {url}")

# Call the test function
test_download_image()
def test_resize_images():
    image_paths = ["./images/image1.jpg", "./images/image2.jpg"]
    output_folder = "./resized_images"
    target_size = (1280, 720)
    resized_paths = resize_images(image_paths, output_folder, target_size)
    print(f"Resized images saved to: {resized_paths}")

# Call the test function
test_resize_images()
