import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os

class CustomCnnModel(nn.Module):
    def __init__(self, input_dim=128, num_classes=3):
        super(CustomCnnModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Compute dynamically the input size for fully connected layers
        self._to_linear = None
        self._get_conv_output(self.input_dim)

        self.fc_layers = nn.Sequential(
            nn.Linear(self._to_linear, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )

    def _get_conv_output(self, input_dim=128):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_dim, input_dim)
            output = self.conv_layers(dummy_input)
            self._to_linear = output.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


class ImageClassifier:
    def __init__(self, model_path, class_name=None):
        self.device = torch.device("cpu")
        self.model = CustomCnnModel(input_dim=128, num_classes=3)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        
        if torch.cuda.is_available():
            try:
                self.device = torch.device("cuda")
                self.model.to(self.device)
            except Exception:
                self.device = torch.device("cpu")

        self.model.eval()

        if class_name is None:
            self.class_name = {0: 'Cat', 1: 'Dog', 2: 'Person'}
        else:
            self.class_name = class_name

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def predict(self, image_input):
        is_path_input = isinstance(image_input, str)
        if is_path_input:
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        current_cuda_available = torch.cuda.is_available()
        target_device = torch.device("cuda" if current_cuda_available else "cpu")
        if self.device != target_device:
            try:
                self.model.to(target_device)
                self.device = target_device
            except Exception:
                self.device = torch.device("cpu")
                self.model.to(self.device)

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            _, predicted = torch.max(output, 1)

        label = self.class_name.get(predicted.item(), f"Class {predicted.item()}")

        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        h, w = img_bgr.shape[:2]
        font_scale = max(0.6, min(w, h) / 300.0)
        thickness = max(1, int(min(w, h) / 150.0))
        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        box_padding = 8
        cv2.rectangle(
            img_bgr,
            (10, 10),
            (10 + text_w + 2 * box_padding, 10 + text_h + 2 * box_padding),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            img_bgr,
            label,
            (10 + box_padding, 10 + text_h + box_padding),
            font,
            font_scale,
            (0, 255, 0),
            thickness
        )

        annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        labeled_image = Image.fromarray(annotated_rgb)

        if is_path_input:
            output_path = os.path.abspath("labeled_image.jpg")
            labeled_image.save(output_path)
            return label, output_path

        return label, labeled_image