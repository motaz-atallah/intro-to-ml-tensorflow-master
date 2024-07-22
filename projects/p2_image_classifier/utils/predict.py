import tensorflow as tf
from utils.preprocess import process_image

def predict(image_path, model, top_k=5):
    # Preprocess the image
    image = process_image(image_path)
        
    # Make predictions
    predictions = model.predict(image)
    # Get the top K predictions
    probs, classes = tf.nn.top_k(predictions, k=top_k)
    
    # Convert the tensors to numpy arrays
    probs = probs.numpy().flatten()
    classes = classes.numpy().flatten()
    
    return probs, classes
