import json
import os
from threading import Thread

from flask import Flask, jsonify, render_template, request

from components.image_data import Images
from utils import hex_to_name, hex_to_rgb

webapp = Flask(__name__)


# For testing
def get_images_data():
    images_data_json = os.getenv(
        "images_data", "[]"
    )  # Default to an empty list if not found
    images_data = json.loads(images_data_json)
    return images_data


@webapp.route("/")
def home():
    images_data = get_images_data()
    # Render a template (you'll create this next)
    for image in images_data:
        image["labels"] = list(image["keywords"].keys())
        image["data"] = list(image["keywords"].values())
    return render_template("index.html", images_data=images_data)


@webapp.route("/results", methods=["POST"])
def results():
    selected_keywords = request.form.getlist("keywords")
    selected_color = request.form.get("color")
    selected_source = request.form.get("source")
    selected_type = request.form.get("type")
    """unique_color_names = []
    for colour in selected_colors:
        # Convert hex to color name
        color_name = hex_to_name(colour)
        unique_color_names.append(color_name)
    """
    # combined_list = unique_color_names + selected_keywords

    # Join into a comma-delimited string

    rgb = None
    if selected_color:
        rgb = hex_to_rgb(selected_color)
        rgb_name = hex_to_name(selected_color)
    print(rgb)
    print(rgb_name)
    search_query = ", ".join(selected_keywords)
    # print(selected_keywords)
    # Process the selected keywords and colors
    # For example, save them to a database or use them in some logic
    # search_query = "blouse"
    imgs = Images()
    if selected_source == "asos":
        if selected_type == "mask":
            image_urls, simple_image_urls = imgs.collect_asos_images(
                search_query, 5, rgb
            )
        else:
            image_urls, simple_image_urls = imgs.collect_asos_images(
                search_query, 1, rgb
            )
    else:
        if selected_type == "mask":
            image_urls = imgs.collect_unsplash_images(search_query, 5, rgb)
        else:
            image_urls = imgs.collect_unsplash_images(search_query, 1, rgb)
        simple_image_urls = image_urls
    print(search_query)
    if selected_type == "mask":
        return render_template("results_page_mask.html", image_urls=simple_image_urls)
    else:
        return render_template(
            "results_page.html", image_urls=image_urls, selected_source=selected_source
        )


@webapp.route("/update-images", methods=["POST"])
def update_images():
    global images_data
    images_data = request.json  # Assuming JSON data
    return jsonify({"status": "success"})


def run_app():
    def run():
        webapp.run(
            debug=True, use_reloader=False
        )  # use_reloader=False is important here

    thread = Thread(target=run)
    thread.start()
