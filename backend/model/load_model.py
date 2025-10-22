import torch
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from torchvision import transforms

from .train_model import Model

class ClassificationModel():
    def __init__(self, model_path: str):
        self.model = Model()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def preprocess(self, encoded_image: str):
        image_data = base64.b64decode(encoded_image.split(",")[1])
        image = Image.open(BytesIO(image_data)).convert("L")

        transform = transforms.Compose([
            transforms.Grayscale(),         
            transforms.Resize((28, 28)),    
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        img_tensor = transform(image).unsqueeze(0)

        return img_tensor

    def predict(self, encoded_image: str):
        input_tensor = self.preprocess(encoded_image)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities
            confidence, predicted_class = torch.max(probabilities, dim=1)  # Get max probability and class

        return predicted_class.item(), confidence.item()