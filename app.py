import gradio as gr
import os
import torch

from model import create_model
from timeit import default_timer as timer
from typing import Tuple, Dict

with open(foodvision_101_class_names_path, "r") as f:
  class_names = [i.strip() for i in  f.readlines()] 

model, auto_transforms = create_model(len(class_names))

model.load_state_dict(
    torch.load(
        f="effnetb2_feature_extractor_food101.pth",
        map_location=torch.device("cpu")
    )
)

def predict(img) -> Tuple[Dict, float]:
  """Transforms and performs a prediction on img and returns prediction and time taken.
  """

  start_time = timer()

  img = auto_transforms(img).unsqueeze(0)

  model.eval()
  with torch.inference_mode():
    pred_probs = torch.softmax(model(img), dim=1)

  pred_labels_and_probs = {class_names[i].title().replace("_", " "): float(pred_probs[0][i]) for i in range(len(class_names))}

  pred_time = round(timer() - start_time, 2)

  return pred_labels_and_probs, pred_time


title = "FoodVision 101"
description = f"CV model classifying 101 classes of food using food101 and EfficientNetB2 feature extractor."
article = "© Amuni Pelumi https://www.amunipelumi.com/"

example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=1, label="Predicted Food"),
                             gr.Number(label="Prediction Duration (S)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

demo.launch()  
