# Image Captioning AI - IBM Coursera Project

This repository contains the code for the Image Captioning AI project from the IBM "AI Applications in Python" course on Coursera. 

The project leverages the **BLIP (Bootstrapping Language-Image Pre-training)** model from **Hugging Face Transformers** to transform visual data into machine-readable natural language. By teaching a computer to look at a picture and describe it, this technology has vast real-world applications.

## 🚀 Features & Scripts

The repository includes several implementations of image captioning for different use cases:

- **`image_captioning_app.py`**: A user-friendly web application built with **Gradio**. It provides an intuitive interface where users can upload an image and instantly receive an AI-generated caption.
- **`generate_web_images_caps.py`**: An automated tool that scrapes a web page (e.g., a Wikipedia article) using BeautifulSoup, extracts all valid images, and uses the BLIP model to generate and save captions for them in a text file.
- **`caption_local_DIR_images.py`**: A batch processing script that reads all images from a local `./images` directory and generates a text file (`captions2.txt`) containing descriptions for every image.
- **`image_cap.py`**: A simple, standalone script demonstrating the basic usage of the BLIP model to caption a single local image.

## 💡 Why Image Captioning?

Image captioning AI makes a significant difference in many areas:
* **Improves accessibility:** Helps visually impaired individuals understand visual content through screen readers.
* **Enhances SEO:** Assists search engines in identifying the content of images (via alt text).
* **Facilitates content discovery:** Enables efficient analysis and categorization of large image databases.
* **Saves time:** Automated captioning is significantly faster and more scalable than manual efforts.

### Real-World Scenario: News & Media
Imagine a news agency publishing hundreds of articles daily. Writing descriptive captions for each image manually is a tedious task. Using this tool, journalists can feed their selected images into the AI to automatically generate draft captions. Once approved, these captions serve as **alt text**—improving accessibility for visually impaired readers and boosting the article's SEO on search engines.

## 🛠️ Technologies Used

- **Python**
- **Hugging Face Transformers** (BLIP model: `Salesforce/blip-image-captioning-base` & `Salesforce/blip2-opt-2.7b`)
- **Gradio** (for the web interface)
- **BeautifulSoup** (for web scraping)
- **PIL (Python Imaging Library)**
- **PyTorch**

## ⚙️ How to Run

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd blip-image-captioning-ai
   ```

2. **Install the dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: You may also need to install PyTorch separately depending on your system configuration).*

3. **Run the Gradio Web App:**
   ```bash
   python image_captioning_app.py
   ```
   This will launch a local server. Open the provided URL in your browser to interact with the AI!

4. **Run the other scripts** depending on your needs:
   ```bash
   python generate_web_images_caps.py
   python caption_local_DIR_images.py
   python image_cap.py
   ```

## 🎓 Learning Objectives Achieved
* Implemented an image captioning tool using the BLIP model from Hugging Face's Transformers.
* Used Gradio to provide a user-friendly interface for an AI application.
* Adapted the AI tool for real-world business scenarios, such as automating alt-text generation for web articles.
