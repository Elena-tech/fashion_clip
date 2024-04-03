import os
import time
from collections import Counter

import numpy as np
import requests
import webcolors
from dotenv import load_dotenv
from PIL import Image
from sklearn.cluster import KMeans

from config import URL

load_dotenv()


def create_folder():
    tmpdir = os.getenv("TMP_DIR")
    # print(tmpdir)
    # create a tmp folder in order to save the resized input image
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)


def get_filelocation(filename):
    tmpdir = os.getenv("TMP_DIR")
    split_tup = os.path.splitext(filename)
    # extract the file name and extension
    file_extension = split_tup[1]

    tmpfilename = time.strftime("%Y%m%d-%H%M%S") + file_extension
    file_dir = "%s/%s" % (tmpdir, tmpfilename)
    return file_dir, tmpfilename


def cleanse_keywords(keywords):

    clothing_related = False
    keywords_to_remove = ["drawing", "clothing"]  # Example keywords to remove

    # List of simple clothing-related keywords for illustration
    clothing_related_keywords = [
        "shirt",
        "t-shirt",
        "polo shirt",
        "tank top",
        "blouse",
        "sweater",
        "cardigan",
        "hoodie",
        "jacket",
        "coat",
        "blazer",
        "trench coat",
        "raincoat",
        "parka",
        "poncho",
        "vest",
        "dress",
        "evening gown",
        "cocktail dress",
        "summer dress",
        "maxi dress",
        "mini dress",
        "wrap dress",
        "skirt",
        "mini skirt",
        "midi skirt",
        "maxi skirt",
        "pleated skirt",
        "pencil skirt",
        "pants",
        "trousers",
        "jeans",
        "leggings",
        "jeggings",
        "cargo pants",
        "sweatpants",
        "shorts",
        "culottes",
        "suit",
        "tuxedo",
        "business suit",
        "tracksuit",
        "jumpsuit",
        "romper",
        "overalls",
        "underwear",
        "briefs",
        "boxers",
        "bra",
        "sports bra",
        "lingerie",
        "panties",
        "thong",
        "socks",
        "stockings",
        "tights",
        "shoes",
        "boots",
        "sneakers",
        "sandals",
        "flats",
        "heels",
        "wedges",
        "loafers",
        "slippers",
        "accessories",
        "belt",
        "hat",
        "cap",
        "beanie",
        "scarf",
        "gloves",
        "mittens",
        "sunglasses",
        "eyeglasses",
        "jewelry",
        "watch",
        "bracelet",
        "necklace",
        "earrings",
        "ring",
        "brooch",
        "cufflinks",
        "bags",
        "backpack",
        "handbag",
        "clutch",
        "tote bag",
        "messenger bag",
        "wallet",
        "briefcase",
    ]
    keywords_list = [keyword.strip() for keyword in keywords.split(",")]
    # Remove duplicates
    unique_keywords = set(keywords_list)
    print(unique_keywords)

    # Filter out unwanted keywords
    filtered_keywords = [kw for kw in unique_keywords if kw not in keywords_to_remove]
    print(filtered_keywords)
    """remove clothing filter
    clothing_keywords = [
        kw for kw in unique_keywords if kw in clothing_related_keywords
    ]
    """
    clothing_keywords = filtered_keywords
    # Check if any of the remaining keywords are clothing-related
    if len(clothing_keywords) > 0:
        clothing_related = True
    print(clothing_keywords)
    # Return the cleansed list of keywords and whether they are clothing-related
    return clothing_keywords, clothing_related


def rgb_to_hex(rgb):
    # Converts an RGB tuple to a hexadecimal string
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def hex_to_rgb(hex_color):
    """Convert a hex color to an RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def hex_to_name(hex_color):
    rgb = hex_to_rgb(hex_color)
    try:
        # Directly find the name if exact match exists
        return webcolors.rgb_to_name(rgb)
    except ValueError:
        # Find the closest color name if no exact match
        return closest_color(rgb)


def get_truly_dominant_colors(filedir, num_colors=5):
    # Load the image and convert it to a numpy array
    image = Image.open(filedir)
    image_np = np.array(image)

    # Check if the image has an alpha channel and remove it if present
    pixels = (
        image_np[..., :3].reshape(-1, 3)
        if image_np.shape[2] == 4
        else image_np.reshape(-1, 3)
    )

    # Perform KMeans clustering to find dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Extract the RGB values of the cluster centers
    dominant_colors = kmeans.cluster_centers_.astype(int)

    # Convert these RGB values back to hexadecimal format
    dominant_hex_colors = [rgb_to_hex(color) for color in dominant_colors]
    seen = set()
    hex_colors_no_duplicates = [
        x for x in dominant_hex_colors if not (x in seen or seen.add(x))
    ]
    return hex_colors_no_duplicates


def get_dominant_colors(filedir, num_colors=400):
    image = Image.open(filedir)
    # Resize the image to speed up processing
    small_image = image.resize((image.width // 10, image.height // 10))
    # Convert the image data to a flat list of RGB values
    data = small_image.getdata()
    # Count the occurrences of each color
    counter = Counter(data)
    # Get the most common colors
    most_common_colors = counter.most_common(num_colors)
    # Convert RGB values to the closest named color
    hex_colors = [rgb_to_hex(rgb) for rgb, _ in most_common_colors]
    seen = set()
    hex_colors_no_duplicates = [x for x in hex_colors if not (x in seen or seen.add(x))]
    hex_colors_no_similarities = filter_similar_colors(hex_colors_no_duplicates)
    return hex_colors_no_similarities


def get_representative_colors(filedir, num_clusters=3):
    image = Image.open(filedir)
    # Resize the image to speed up processing
    small_image = image.resize((image.width // 10, image.height // 10))
    # Convert the image data to a numpy array of RGB values
    data = np.array(small_image.getdata())

    # Use K-means clustering to find clusters of colors
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)

    # Get the RGB values of the cluster centroids
    centroids = kmeans.cluster_centers_

    # Convert centroids to integers from float (since RGB values should be integers)
    centroids = centroids.astype(int)

    # Convert each centroid to a hex color
    hex_colors = [rgb_to_hex(tuple(rgb)) for rgb in centroids]

    # Remove duplicates
    hex_colors = list(set(hex_colors))

    return hex_colors


def is_flask_running():
    url = URL
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
    except requests.ConnectionError:
        return False
    return False


def closest_color_text(rgb_color):
    colors = {
        "Black": (0, 0, 0),
        "White": (255, 255, 255),
        "Gray": (128, 128, 128),
        "Silver": (192, 192, 192),
        "Red": (255, 0, 0),
        "Maroon": (128, 0, 0),
        "Rose": (255, 102, 204),
        "Pink": (255, 192, 203),
        "Orange": (255, 165, 0),
        "Coral": (255, 127, 80),
        "Gold": (255, 215, 0),
        "Yellow": (255, 255, 0),
        "Cream": (255, 253, 208),
        "Lime Green": (50, 205, 50),
        "Green": (0, 128, 0),
        "Olive": (128, 128, 0),
        "Turquoise": (64, 224, 208),
        "Teal": (0, 128, 128),
        "Light Blue": (173, 216, 230),
        "Sky Blue": (135, 206, 235),
        "Blue": (0, 0, 255),
        "Navy": (0, 0, 128),
        "Lavender": (230, 230, 250),
        "Purple": (128, 0, 128),
        "Violet": (238, 130, 238),
        "Magenta": (255, 0, 255),
        "Salmon": (250, 128, 114),
        "Beige": (245, 245, 220),
        "Tan": (210, 180, 140),
        "Brown": (165, 42, 42),
        "Burgundy": (128, 0, 32),
        "Khaki": (195, 176, 145),
        "Emerald Green": (0, 128, 0),
        "Mint Green": (152, 255, 152),
        "Mustard": (255, 219, 88),
        "Peach": (255, 229, 180),
        "Rust": (183, 65, 14),
        "Charcoal": (54, 69, 79),
        "Ivory": (255, 255, 240),
        "Plum": (221, 160, 221),
    }

    min_distance = float("inf")
    closest_color_name = None
    for color_name, color_rgb in colors.items():
        distance = sum((a - b) ** 2 for a, b in zip(rgb_color, color_rgb)) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_color_name = color_name

    return closest_color_name


def rgb_distance(rgb1, rgb2):
    """Calculate the Euclidean distance between two RGB colors."""
    return sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5


def filter_similar_colors(hex_colors, threshold=100):
    """Filter out similar colors from a list of hex colors based on RGB distance."""
    filtered_colors = []
    for current_hex in hex_colors:
        current_rgb = hex_to_rgb(current_hex)
        if all(
            rgb_distance(current_rgb, hex_to_rgb(other_hex)) > threshold
            for other_hex in filtered_colors
        ):
            filtered_colors.append(current_hex)
    return filtered_colors
