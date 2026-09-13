from io import BytesIO
from bs4 import BeautifulSoup
from PIL import Image
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# URL of the page to scrape
url = "https://en.wikipedia.org/wiki/IBM"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

try:
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    print("Status:", response.status_code, "HTML size:", len(response.text))
except requests.exceptions.RequestException as e:
    print(f"Failed to retrieve the main page: {e}")
    exit()

soup = BeautifulSoup(response.text, "html.parser")
img_elements = soup.find_all("img")
print(f"Found {len(img_elements)} <img> tags")

with open("captions.txt", "w", encoding="utf-8") as caption_file:
    for idx, img_element in enumerate(img_elements, start=1):
        # Try different attributes
        img_url = img_element.get("src") or img_element.get("data-src")
        if not img_url and img_element.has_attr("srcset"):
            img_url = img_element["srcset"].split()[0]

        if not img_url:
            continue

        # Skip SVGs directly
        if img_url.endswith(".svg") or ".svg" in img_url:
            continue

        # Fix relative URLs
        if img_url.startswith("//"):
            img_url = "https:" + img_url
        elif img_url.startswith("/"):
            img_url = "https://en.wikipedia.org" + img_url
        elif not img_url.startswith("http"):
            continue

        try:
            r = requests.get(img_url, timeout=10, headers=headers)
            r.raise_for_status()

            raw_image = Image.open(BytesIO(r.content))

            # Skip very small images (e.g., icons, spacers)
            if raw_image.size[0] * raw_image.size[1] < 200:
                continue

            raw_image = raw_image.convert("RGB")

            # Process the image with a text prompt
            text = "the image of"
            inputs = processor(
                images=raw_image, text=text, return_tensors="pt"
            )

            # Generate caption using modern HF parameters
            out = model.generate(
                **inputs,
                max_new_tokens=50,
            )
            caption = processor.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

            caption_file.write(f"{img_url}: {caption}\n")
            print(f"[{idx}] Caption saved")

        except requests.exceptions.RequestException as e:
            print(f"[{idx}] Network error fetching image: {e}")
            continue
        except (OSError, SyntaxError) as e:
            # Skip images PIL cannot open (corrupt, unsupported webp/gif configs)
            print(f"[{idx}] PIL could not open image: {e}")
            continue
        except Exception as e:
            print(f"[{idx}] Error processing image: {e}")
            continue
