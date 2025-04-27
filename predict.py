# Purpose: A script for loading new images and making predictions.

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from utils import classes

def image_prediction(model, img_path, img_height=256, img_width=256):
    img = load_img(img_path, target_size=(img_height, img_width))
    img_arr = img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    prediction = model.predict(img_arr)
    return classes[np.argmax(prediction)]

if __name__ == "__main__":
    model = load_model('best_model.h5')
    img_path = "/path/to/your/image.png"
    prediction = image_prediction(model, img_path)
    print(f"Predicted class: {prediction}")
