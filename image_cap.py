import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

#Load and process image
img_path = "./images/lion.jpg"
# convert it into an RGB format 
image = Image.open(img_path).convert('RGB')

# You do not need a question for image captioning
text = "a picture of"
inputs = processor(images=image, text=text, return_tensors="pt")

print("\ninputs: ", inputs)
# Generate a caption for the image
outputs = model.generate(**inputs, max_length=50)
print("\noutputs: ", outputs)

# Decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)

# Print the caption
print(caption)


