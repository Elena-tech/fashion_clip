import json
import os
import tkinter as tk
import webbrowser
from tkinter import messagebox

import matplotlib.pyplot as plt
import requests
import torch
from matplotlib.widgets import Button
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.drawing import DrawingApp
from components.aws import RunRekognition
from components.image_data import Images
from config import DISPLAY_CHART_GUI, DISPLAY_RESULTS_GUI, URL
from utils import (
    cleanse_keywords,
    get_truly_dominant_colors,
    hex_to_name,
    is_flask_running,
)
from webapp import run_app

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

# Can change in config.py
display_chart_gui = DISPLAY_CHART_GUI
display_results_gui = DISPLAY_RESULTS_GUI
home_url = URL


def close_event(event):
    plt.close()  # This will close the figure window


def display_clip_results(images, probs, classes):

    fig = plt.figure(figsize=(8, 20))

    # Adjust the subplot to make space for the close button
    ax = fig.add_axes([0.1, 0.01, 0.8, 0.05])  # x, y, width, height

    button = Button(ax, "Close")
    button.on_clicked(close_event)

    for idx in range(len(images)):

        # show original image
        fig.add_subplot(len(images), 2, 2 * (idx + 1) - 1)
        plt.imshow(images[idx])
        plt.xticks([])
        plt.yticks([])

        # show probabilities
        fig.add_subplot(len(images), 2, 2 * (idx + 1))
        plt.barh(
            range(len(probs[0].detach().numpy())),
            probs[idx].detach().numpy(),
            tick_label=classes,
        )
        plt.xlim(0, 1.0)

        plt.subplots_adjust(
            left=0.1, bottom=0.2, right=0.9, top=0.9, wspace=0.2, hspace=0.8
        )

    plt.show()

    # After displaying, ask user for input


def image_grid(imgs, cols):
    rows = (len(imgs) + cols - 1) // cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def get_top_keywords(probs, keywords, num_top_keywords=10):
    if not isinstance(probs, torch.Tensor):
        probs = torch.tensor(probs)

    # Ensure probabilities tensor is float for torch.topk
    probs = probs.float()

    # Determine the number of top keywords to retrieve, which cannot exceed the length of keywords
    num_top_keywords = min(num_top_keywords, len(keywords))

    # Get the highest probabilities and their indices
    top_probs, top_idxs = torch.topk(probs, k=num_top_keywords)

    # Convert indices and probabilities to lists
    top_idxs = top_idxs.tolist()
    top_probs = top_probs.tolist()
    flat_top_idxs = top_idxs[0] if top_idxs else []
    flat_top_probs = top_probs[0] if top_probs else []
    # print(flat_top_idxs)
    # print(flat_top_probs)
    # Flatten top_idxs if it's a list of lists
    # if top_idxs and isinstance(top_idxs[0], list):
    #    top_idxs = [item for sublist in top_idxs for item in sublist]

    # Create a dictionary of top keywords and their probabilities

    top_keywords_probs = {}  # Start with an empty dictionary

    for key, value in zip(flat_top_idxs, flat_top_probs):
        actualkey = keywords[key]
        print(actualkey)
        top_keywords_probs[actualkey] = value

    return top_keywords_probs


def check_image_using_clip(filedir, keywords):
    images = []
    images.append(Image.open(filedir).convert("RGB"))

    inputs = processor(
        text=keywords,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    print(probs)
    if display_chart_gui:
        display_clip_results(images, probs, keywords)

    return get_top_keywords(probs, keywords)


def show_grid(request):

    imgs = Images()
    image_urls = imgs.collect_unsplash_images(request, 1)
    images = []
    for url in image_urls:
        print(url)
        images.append(Image.open(requests.get(url, stream=True).raw))

    grid = image_grid(images, cols=3)
    grid.show()


def image_saved_callback(filedir, filename):
    aws = RunRekognition()
    image_uri = aws.upload_to_bucket(filename, filedir)
    # print(f"Image saved: {filedir}")
    print(f"Image saved on AWS: {image_uri}")
    keywords = aws.get_labels(filedir)
    textfromimage = aws.get_text(filedir)
    if textfromimage is not None:
        keywords = keywords + ", " + textfromimage
    # print(keywords)
    unique_clothing_keywords, isclothing = cleanse_keywords(keywords)
    if not isclothing:
        messagebox.showerror(
            "Not Clothing-Related",
            "The image you saved does not seem to be related to clothing. Please try again.",
        )
    else:
        # Now check on Clip & get top keywords
        top_keywords = check_image_using_clip(filedir, unique_clothing_keywords)
        print(top_keywords)
        # Now let display the web
        # colours = get_representative_colors(filedir)
        colours = get_truly_dominant_colors(filedir)
        print(colours)
        images_data = [
            {
                "image_path": filedir.replace("static/", "", 1),
                "keywords": top_keywords,
                "colors": colours,  # RGB format
            },
            # Add more image data as needed
        ]

        # colours = get_dominant_colors(filedir)
        print(images_data)
        # Build search string
        request = None
        filtered_keys = [key for key in top_keywords.keys() if key][:2]
        request = ", ".join(filtered_keys)
        request += ","
        unique_color_names = []
        seen = set()

        for colour in colours:
            # Convert hex to color name
            color_name = hex_to_name(colour)

            # Check if the color name has already been seen
            if color_name not in seen:
                seen.add(color_name)  # Mark as seen
                unique_color_names.append(color_name)

        request += ",".join(unique_color_names)

        print(request)
        if display_results_gui:
            show_grid(request)
        images_data_json = json.dumps(images_data)
        os.environ["images_data"] = images_data_json
        print(os.environ["images_data"])
        if not is_flask_running():
            run_app()
        else:
            webbrowser.open(home_url)


def main():
    root = tk.Tk()
    app = DrawingApp(root, save_callback=image_saved_callback)
    root.mainloop()


if __name__ == "__main__":
    main()
