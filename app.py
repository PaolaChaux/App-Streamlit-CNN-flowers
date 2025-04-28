import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# ------------ Definir modelo CNN ------------

class SimpleCNN(nn.Module):
    def __init__(self, in_chans=3, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.net(x)

# ------------ Función para cargar modelo ------------

def load_model(model_class, path='model_cnn.pth', device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# ------------ Transformación para la imagen ------------

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ------------ Diccionario índice a nombre de clase ------------

idx_to_class = {
    0: 'daisy',
    1: 'dandelion',
    2: 'rose',
    3: 'sunflower',
    4: 'tulip'
}

# ------------ Cargar modelo ------------

model = load_model(lambda: SimpleCNN(in_chans=3, n_classes=5))

# ------------ App Streamlit ------------

st.title("🌸 Clasificador de Flores con CNN 🌻")

uploaded_file = st.file_uploader("Sube una imagen de flor", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True)
    st.write("Clasificando...")

    # Preprocesar imagen
    input_tensor = transform(image).unsqueeze(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_tensor = input_tensor.to(device)
    model = model.to(device)

    # Hacer predicción
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)  # Softmax para probabilidades
        pred_idx = probabilities.argmax(dim=1).item()
        pred_label = idx_to_class[pred_idx]
        confidence = probabilities[0, pred_idx].item() * 100  # Convertir a %

    st.success(f"**Predicción:** {pred_label} 🌼")
    st.info(f"**Confianza:** {confidence:.2f}%")

    # Mostrar tabla de todas las probabilidades
    st.subheader("Todas las probabilidades:")
    prob_dict = {idx_to_class[i]: float(probabilities[0, i]) for i in range(len(idx_to_class))}
    st.dataframe(prob_dict)
