import os
import xml.etree.ElementTree as ET
import requests
import tarfile
import shutil

images_path = './IU-XRay/images'
reports_path = './IU-XRay/reports'

# Create directories if not exist
os.makedirs(images_path, exist_ok=True)
os.makedirs(reports_path, exist_ok=True)

# Helper function to download a file
def download_file(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading {url} ...")
        response = requests.get(url, stream=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    else:
        print(f"{save_path} already exists")

# Download files
download_file("https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz",
              os.path.join(images_path, "images.tgz"))
download_file("https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz",
              os.path.join(reports_path, "reports.tgz"))

# Extract tar.gz files
def extract_tgz(tgz_path, extract_to):
    print(f"Extracting {tgz_path} ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=extract_to)

extract_tgz(os.path.join(images_path, "images.tgz"), images_path)
extract_tgz(os.path.join(reports_path, "reports.tgz"), reports_path)

# Move XML files from extracted folder to reports_path
extracted_folder = os.path.join(reports_path, "ecgen-radiology")
if os.path.exists(extracted_folder):
    for file in os.listdir(extracted_folder):
        if file.endswith(".xml"):
            shutil.move(os.path.join(extracted_folder, file), reports_path)
    shutil.rmtree(extracted_folder)

# Remove the downloaded tgz files
os.remove(os.path.join(images_path, "images.tgz"))
os.remove(os.path.join(reports_path, "reports.tgz"))

# Process reports
reports = [f for f in os.listdir(reports_path) if f.endswith(".xml")]
reports.sort()

reports_with_no_image = []
reports_with_empty_sections = []
reports_with_no_impression = []
reports_with_no_findings = []

images_captions = {}
reports_with_images = {}
text_of_reports = {}

for report in reports:
    tree = ET.parse(os.path.join(reports_path, report))
    root = tree.getroot()
    img_ids = []

    images = root.findall("parentImage")
    if len(images) == 0:
        reports_with_no_image.append(report)
        continue

    impression = None
    findings = None
    sections = root.find("MedlineCitation").find("Article").find("Abstract").findall("AbstractText")
    for section in sections:
        if section.get("Label") == "FINDINGS":
            findings = section.text
        elif section.get("Label") == "IMPRESSION":
            impression = section.text

    if impression is None and findings is None:
        reports_with_empty_sections.append(report)
        continue
    elif impression is None:
        reports_with_no_impression.append(report)
        caption = findings
    elif findings is None:
        reports_with_no_findings.append(report)
        caption = impression
    else:
        caption = impression + " " + findings

    for image in images:
        images_captions[image.get("id") + ".png"] = caption
        img_ids.append(image.get("id") + ".png")

    reports_with_images[report] = img_ids
    text_of_reports[report] = caption

print("Found", len(reports_with_no_image), "reports with no associated image")
print("Found", len(reports_with_empty_sections), "reports with empty Impression and Findings sections")
print("Found", len(reports_with_no_impression), "reports with no Impression section")
print("Found", len(reports_with_no_findings), "reports with no Findings section")
print("Collected", len(images_captions), "image-caption pairs")
