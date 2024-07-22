import tensorflow as tf
import numpy as np

from PIL import Image

image_size = (224, 224)

def process_image(image_path):
    # Preprocess the image
    image = Image.open(image_path)
    image = np.asarray(image)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, image_size)    
    image = image / 255.0
    processed_image = image.numpy()
    # Add extra dimension
    return np.expand_dims(processed_image, axis=0)
