from datetime import datetime
from urllib.request import urlopen

import cv2
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
from PIL import Image
from skimage import io
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import applications


class umap_plotting:
    def __init__(self, return_images):
        self.return_images = return_images

    def plot_umap(self, feature_type, features, n_neighbors, min_dist):

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)

        pipe = Pipeline([("scaling", StandardScaler()), ("umap", reducer)])
        embedding = pipe.fit_transform(features)

        tx, ty = embedding[:, 0], embedding[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

        width = 4000
        height = 3000
        max_dim = 100

        full_image = Image.new("RGBA", (width, height))
        for img, x, y in zip(self.return_images, tx, ty):
            tile = Image.open(urlopen(img))
            rs = max(1, tile.width / max_dim, tile.height / max_dim)
            tile = tile.resize(
                (int(tile.width / rs), int(tile.height / rs)), Image.Resampling.LANCZOS
            )
            full_image.paste(
                tile,
                (int((width - max_dim) * x), int((height - max_dim) * y)),
                mask=tile.convert("RGBA"),
            )

        plt.figure(figsize=(16, 12))
        full_image.save(
            "images/UMAP_"
            + feature_type
            + "_nn="
            + str(n_neighbors)
            + "_md="
            + str(min_dist)
            + "_"
            + str(datetime.now().strftime("%Y%m%d-%H%M%S"))
            + ".png"
        )
        return plt.imshow(full_image)

    def get_imagenet_features(image_data):
        image_data = [cv2.resize(io.imread(image), (224, 224)) for image in image_data]
        image_data = np.float32(image_data)
        image_data /= 255
        image_data = np.array(image_data)

        model = applications.MobileNetV2(
            input_shape=(224, 224, 3), alpha=1.0, weights="imagenet", pooling=None
        )
        pred = model.predict(image_data)
        imagenet_features = pred.reshape(image_data.shape[0], -1)

        return imagenet_features
