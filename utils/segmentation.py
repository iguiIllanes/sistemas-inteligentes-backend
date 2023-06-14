import cv2
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import requests
from io import BytesIO

class Segmentation:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def convert_image_to_array(self, image):
        # image = Image.open(BytesIO(image)) #TODO: remove this
        resized_image = image.resize((150, 150))
        rgb_image = resized_image.convert('RGB')
        image_array = np.array(rgb_image)
        image_array = image_array / 255
        return image_array

    def get_segmented_image(self,image):
        image = Image.open(BytesIO(image))
        image = np.array(image)/255
        image_reshaped = image.reshape(-1, image.shape[-1])

        k=2

        kmeans = KMeans(n_clusters=k).fit(image_reshaped)

        segmented_img = kmeans.cluster_centers_[kmeans.labels_]
        segmented_img = segmented_img.reshape(image.shape)

        img_gray = cv2.cvtColor((segmented_img*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(img_gray, threshold1=30, threshold2=100)

        fig, axs = plt.subplots(1, 3, figsize=(20,12))
        axs[0].imshow(image)
        axs[0].set_title('Imagen Original')

        axs[1].imshow(segmented_img)
        axs[1].set_title('Imagen Segmentada')

        axs[2].imshow(edges, cmap='gray')
        axs[2].set_title('Bordes Detectados')

        for ax in axs:
            ax.axis('off')

        plt.savefig('segmented_image.png')

        return segmented_img

    def make_prediction(self, image):
        # image_array = self.get_segmented_image(image)
        image_array = self.convert_image_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        prediction = self.model.predict(image_array)
        if prediction[0][0] > 0.5:
            return 'es un perro'
        else:
            return 'es un gato'





# segmentation = Segmentation('models/dogs_cats.h5')
#
# url_imagen = 'https://c.files.bbci.co.uk/48DD/production/_107435681_perro1.jpg'
# response = requests.get(url_imagen)
#
# print(segmentation.make_prediction(response.content))
#
# # segmentation.get_segmented_image(response.content)


