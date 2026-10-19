try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is not None:
                return func
            def decorator(f):
                return f
            return decorator

import os
import sys
from PIL import Image
import gradio as gr

core_dir = os.path.dirname(os.path.abspath(__file__))
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

from predict import ImageClassifier

# Dynamically locate the model weights
project_dir = os.path.dirname(core_dir)
candidate_paths = [
    os.path.join(project_dir, "model", "cnn_128_model-100.pth"),
    os.path.join(core_dir, "model", "cnn_128_model-100.pth"),
    os.path.join(core_dir, "cnn_128_model-100.pth"),
]

model_path = None
for candidate in candidate_paths:
    if os.path.exists(candidate):
        model_path = candidate
        break

if not model_path:
    raise FileNotFoundError("Could not find 'cnn_128_model-100.pth' in any standard location.")

class_name = {0: 'Cat', 1: 'Dog', 2: 'Person'}
classifier = ImageClassifier(model_path=model_path, class_name=class_name)


@spaces.GPU
def classify_image(image):
    if image is None:
        return "Please upload an image.", None

    label, labeled_img = classifier.predict(image)
    return label, labeled_img


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type='pil', label="Upload Image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Image(label="Labeled Image")
    ],
    title="Image Classification Gradio App",
    description="Upload an image to classify it as Dog, Cat, or Person."
)

if __name__ == "__main__":
    demo.launch()