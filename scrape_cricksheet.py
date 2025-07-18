import os
import requests
from bs4 import BeautifulSoup
from zipfile import ZipFile

# Configuration
base_url = "https://cricsheet.org"
downloads_url = f"{base_url}/downloads/"
# Mapping filenames to subfolder destinations
file_destinations = {
    "odis_csv2.zip": r"D:/GUVI_Second_Project/ODI_Match/odis_csv2",
    "t20s_json.zip": r"D:/GUVI_Second_Project/T20_Match/t20s_json",
    "tests_json.zip": r"D:/GUVI_Second_Project/Test_Match/tests_json"
}

# Ensuring folders exist
for path in file_destinations.values():
    os.makedirs(path, exist_ok=True)

# Getting download links
print("Fetching download page now")
response = requests.get(downloads_url)
soup = BeautifulSoup(response.text, "html.parser")

download_links = {}
for a in soup.find_all("a", href=True):
    href = a["href"]
    filename = os.path.basename(href)
    if filename in file_destinations:
        download_links[filename] = base_url + href

# Downloading and extracting
for filename, url in download_links.items():
    extract_dir = file_destinations[filename]
    zip_path = os.path.join(extract_dir, filename)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded: {filename}")

    # Extracting into subfolder
    print(f"Extracting {filename} into {extract_dir} ...")
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted: {filename}")

    # Cleaning up .zip file
    os.remove(zip_path)

print("All files have been downloaded and extracted into proper subfolders.")
