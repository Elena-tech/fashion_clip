import os

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from skimage import color, io
from sklearn.cluster import KMeans

from utils import closest_color_text


class Images:
    def __init__(self):
        load_dotenv()
        self.splash_client_id = os.getenv("CLIENT_ID")  # Getting a key and printing it
        self.splash_api_url = os.getenv("API_URL")  # Getting a key and printing it
        self.asos_api_url = os.getenv("ASOS_BASE_URL")
        self.asos_api_key = os.getenv("ASOS_RAPIDAPI_KEY")
        self.asos_api_host = os.getenv("ASOS_RAPIDAPI_HOST")
        self.images = []

    def collect_unsplash_images(self, query, max_pages=20, rgb=None):
        images = []
        page = 1
        text_colour = None
        print(rgb)
        if rgb is not None:
            text_colour = closest_color_text(rgb)
        new_query = query
        if text_colour is not None:
            new_query = text_colour + " " + query

        print(new_query)
        while page <= max_pages:
            params = {
                "query": new_query,
                "client_id": self.splash_client_id,
                "per_page": 30,
                "page": page,
            }

            # if rgb is not None:
            #    params["color"] = closest_color_text(rgb)
            print(params)
            response = requests.get(self.splash_api_url, params=params)
            if response.status_code == 200:
                data = response.json()["results"]
                if not data:
                    break  # No more images available
                for item in data:
                    images.append(
                        item["urls"]["small"]
                    )  # So we don't need to resize it in the future

            else:
                print(
                    f"Failed to fetch data from Unsplash: HTTP {response.status_code}"
                )
                break
            page += 1
        print(f"Finished collecting {len(images)} images.")
        return images

    def resize_image(self, image):
        height, width = image.shape[:2]
        max_height = 200
        max_width = 200

        # Only shrink if img is bigger than required
        if max_height < height or max_width < width:
            # Get scaling factor
            scaling_factor = max_height / float(height)
            if max_width / float(width) < scaling_factor:
                scaling_factor = max_width / float(width)
            # Resize image
            resized_image = cv2.resize(
                image,
                None,
                fx=scaling_factor,
                fy=scaling_factor,
                interpolation=cv2.INTER_AREA,
            )
            return resized_image
        return image  # Return the original image if no resizing is needed

    def get_colour_palettes(self, image_data):
        colour_palettes = []

        for image_url in image_data:
            image = io.imread(image_url)
            image = self.resize_image(image)
            image = image.reshape((image.shape[0] * image.shape[1], 3))

            kmeans = KMeans(n_clusters=5).fit(image)
            labels = list(kmeans.labels_)
            centroid = kmeans.cluster_centers_

            percent = []
            for i in range(len(centroid)):
                j = labels.count(i)
                j = j / len(labels)
                percent.append(j)

            rgb = np.array(centroid / 255)

            percent_idx = np.array(percent).argsort()
            rgb_sorted = rgb[percent_idx[::-1]]

            colour_palettes.append(rgb_sorted)

        lab = color.rgb2lab(np.array(colour_palettes))
        X = [lab_palette.flatten() for lab_palette in lab]

        return X

    def collect_asos_images(self, query, max_pages=20, rgb=None):
        images = []
        simple_image_urls = []
        page = 1
        text_colour = None
        print(rgb)
        if rgb is not None:
            text_colour = closest_color_text(rgb)
        new_query = query
        if text_colour is not None:
            new_query = text_colour + " " + query

        print(new_query)
        base_url = self.asos_api_url
        headers = {
            "X-RapidAPI-Key": self.asos_api_key,
            "X-RapidAPI-Host": self.asos_api_host,
        }
        # print(base_url)
        # print(headers)
        while page <= max_pages:
            params = {
                "q": new_query,
                "store": "US",
                "offset": "0",
                "limit": "10",
                "country": "US",
                "page": page,
                "sort": "freshness",
                "currency": "USD",
                "sizeSchema": "US",
                "lang": "en-US",
            }
            try:
                response = requests.get(base_url, headers=headers, params=params)
                response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
            except requests.exceptions.HTTPError as errh:
                print("HTTP Error:", errh)
            except requests.exceptions.ConnectionError as errc:
                print("Error Connecting:", errc)
            except requests.exceptions.Timeout as errt:
                print("Timeout Error:", errt)
            except requests.exceptions.RequestException as err:
                print("Oops: Something Else", err)
            else:

                if response.status_code == 200:
                    json_response = response.json()
                    data = json_response["data"]["products"]
                    if not data:
                        break  # No more images available
                    for item in data:
                        image_url = "https://" + item.get("imageUrl")
                        site_url = "https://asos.com/" + item.get("url")
                        images.append({"image_url": image_url, "site_url": site_url})
                        simple_image_urls.append(image_url)
                else:
                    print(
                        f"Failed to fetch data from Unsplash: HTTP {response.status_code}"
                    )
                    break
            page += 1

        print(f"Finished collecting {len(images)} images.")
        return images, simple_image_urls
