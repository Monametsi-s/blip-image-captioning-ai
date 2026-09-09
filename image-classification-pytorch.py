import torch
import gradio as gr
import requests
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# 1. Initialize Model
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()

# 2. Fetch Labels safely (Fixed syntax error)
url = "https://raw.githubusercontent.com/gradio-app/mobilenet-example/master/labels.txt"
try:
    response = requests.get(url, timeout=5)
    labels = [l.strip() for l in response.text.split("\n") if l.strip()]
except requests.RequestException:
    labels = [f"Class {i}" for i in range(1000)]  # Fallback

# 3. Standard Image Preprocessing 
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225]
    )
])

# 4. Optimized Inference Function
def predict(inp_image):
    if inp_image is None:
        return None
        
    # Preprocess and add batch dimension
    inp_image = transform(inp_image).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(inp_image)[0]
        
        # Optimize: Get top 5 indices first, then calculate softmax
        top5_prob, top5_catid = torch.topk(torch.nn.functional.softmax(logits, dim=0), 5)
        
    # Build dictionary containing only the top 5 classes
    confidences = {
        labels[top5_catid[i]]: float(top5_prob[i]) 
        for i in range(5)
    }
    return confidences

# 5. Launch Interface (Fixed syntax errors)
gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="pil"), 
    outputs=gr.Label(num_top_classes=5),
    examples=["./images/lion.jpg", "./images/image5.jpg"]
).launch()