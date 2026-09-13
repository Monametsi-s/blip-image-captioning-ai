"""Module for creating an image captioning app using Gradio."""
import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

#Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    """Function to generate a caption for an image using the BLIP model."""
    try:
        # Convert numpy array to PIL image
        raw_image = Image.fromarray(input_image).convert('RGB')
        text = "a picture of"

        #Process the image and text
        inputs = processor(images=raw_image, text=text, return_tensors="pt")

        # Generate a caption
        outputs = model.generate(**inputs, max_length=50)

        # Decode the generated tokens to text
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"

iface = gr.Interface(
    fn=caption_image,
    inputs="image",
    outputs="text",
    title="Image Captioning App using BLIP",
    description="Upload an image and get a caption for it using the BLIP model."
)

iface.launch(server_name="[IP_ADDRESS]", server_port=3000)