import os
import glob
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Load the pretrained processor and model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# Specify the directory where your images are
image_dir = "./images"
image_exts = ["jpg", "jpeg", "png"]

# Open a file to write the captions
with open("captions2.txt", "w") as caption_file:
    # Iterate over each image file in the directory
    for image_ext in image_exts:
        search_pattern = os.path.join(image_dir, f"*.{image_ext}")
        
        for img_path in glob.glob(search_pattern):
            # Load your image
            raw_image = Image.open(img_path).convert("RGB")
            
            # Prepare image inputs for the model
            inputs = processor(raw_image, return_tensors="pt")
            
            # Generate a caption for the image
            out = model.generate(**inputs, max_new_tokens=50)
            
            # Decode the generated tokens to text
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            # Write the filename and caption to the file
            filename = os.path.basename(img_path)
            caption_file.write(f"{filename}: {caption}\n")
