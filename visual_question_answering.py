import requests
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

# Load VQA model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-vqa-base"
)
model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
)

# Load image
image = Image.open("./images/image3.jpg").convert("RGB")

questions = [
    "What type of tree is in the image?",
    "What color is the tree?"
]

for question in questions:
    inputs = processor(
        image,
        question,
        return_tensors="pt"
    )

    outputs = model.generate(**inputs)
    answer = processor.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print(f"Question: {question}")
    print(f"Answer: {answer}\n")