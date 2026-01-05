import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# -------------------------------
# Step 1: Streamlit page setup
# -------------------------------
st.set_page_config(
    page_title="VisionNet-CPU",
    layout="centered"
)

st.title("🦖VisionNet-CPU: Web-Based Image Classification Using Pretrained ResNet18")
st.write("This application performs image recognition using a pretrained ResNet18 model on CPU.")

# -------------------------------
# Step 3: Force CPU device
# -------------------------------
device = torch.device("cpu")
st.write(f"Running on device: {device}")

# -------------------------------
# Step 4: Load pretrained ResNet18
# -------------------------------
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()
model.to(device)

# -------------------------------
# Step 5: Image preprocessing
# -------------------------------
preprocess = weights.transforms()

# Class labels
class_names = weights.meta["categories"]

# -------------------------------
# Step 6: Upload image interface
# -------------------------------
uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Step 7: Convert image and inference
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Processing image...")

    img_tensor = preprocess(image).unsqueeze(0).to(device)

    # No gradient computation
    with torch.no_grad():
        output = model(img_tensor)

    # Step 8: Softmax + Top-5 predictions
    probabilities = F.softmax(output[0], dim=0)

    top5_prob, top5_catid = torch.topk(probabilities, 5)

    st.subheader("Top-5 Predictions")
    results = []

    for i in range(5):
        results.append([class_names[top5_catid[i]], float(top5_prob[i])])

    df = pd.DataFrame(results, columns=["Class", "Probability"])
    st.table(df)

    # Step 9: Bar chart
    st.subheader("Prediction Probabilities")
    st.bar_chart(df.set_index("Class"))

# Step 10: Automatic Discussion Section
st.subheader("Discussion of Results")

if uploaded_file is not None:

    top1_label = class_names[top5_catid[0]]
    top1_conf = float(top5_prob[0])

    # Generate automatic interpretation
    discussion = ""

    # Confidence level analysis
    if top1_conf > 0.85:
        discussion += f"The model is highly confident that the image belongs to the class **'{top1_label}'** with a probability of {top1_conf:.2f}. "
    elif top1_conf > 0.60:
        discussion += f"The model is moderately confident that the image belongs to the class **'{top1_label}'** with a probability of {top1_conf:.2f}. "
    else:
        discussion += f"The model is uncertain. The highest predicted class is **'{top1_label}'** with a probability of only {top1_conf:.2f}. This suggests overlapping features with other classes. "

    # Spread of top-5 probabilities
    prob_spread = float(top5_prob[0] - top5_prob[4])

    if prob_spread > 0.50:
        discussion += "The probability gap between the top and bottom predictions is large, showing clear separation between classes. "
    else:
        discussion += "The probability values of the top-5 classes are close, indicating the image contains features shared across multiple categories. "

    # General statement about bar chart
    discussion += "The bar chart visualises the confidence distribution across the top-5 predicted classes. Higher bars represent stronger confidence scores produced by the softmax layer."

    st.write(discussion)

else:
    st.write("Upload an image to automatically generate a discussion of the classification result.")




