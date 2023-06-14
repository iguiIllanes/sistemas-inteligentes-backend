from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests


class Classifier:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def convert_image_to_array(self, image):
        resized_image = image.resize((100, 100))
        rgb_image = resized_image.convert('RGB')
        image_array = np.array(rgb_image)
        image_array = image_array / 255
        return image_array

    # def make_graphic(self, image):
    #    # make plt graphic
    #     plt.imshow(image)
    #     plt.save('test.png')

    def make_prediction(self, image):
        image_array = self.convert_image_to_array(image)
        # make_graphic(image_array)
        plt.imshow(image_array)
        plt.savefig('test.png')
        image_array = np.expand_dims(image_array, axis=0)
        prediction = self.model.predict(image_array)
        print('Prediccion: ',prediction[0][0])
        return round(prediction[0][0], 5)


