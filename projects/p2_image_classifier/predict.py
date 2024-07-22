import argparse
import os

from utils.model import load_model, load_class_names
from utils.predict import predict

class Predictor:
    def __init__(self):
        self.args = self.parse_arguments()

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Predict the class of an input image using a trained model.')
        parser.add_argument('image_path', type=str, help='Path to the input image.')
        parser.add_argument('model_path', type=str, help='Path to the saved Keras model.')
        parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes.')
        parser.add_argument('--category_names', type=str, help='Path to the JSON file mapping labels to names.')
        return parser.parse_args()

    def validate_paths(self):
        self.validate_path(self.args.image_path, f'Error: The image path {self.args.image_path} does not exist.')
        self.validate_path(self.args.model_path, f'Error: The model path {self.args.model_path} does not exist.')
        if self.args.category_names:
            self.validate_path(self.args.category_names, f'Error: The category names path {self.args.category_names} does not exist.')
        
    def validate_path(self, file_path, message):
        if not os.path.exists(file_path):
            raise FileNotFoundError(message)

    def run(self):
        self.validate_paths()

        # Load the model
        model = load_model(self.args.model_path)

        # Make predictions
        probs, classes = predict(self.args.image_path, model, self.args.top_k)
        
        print('Probabilities:', probs)
        
        # Load class names if provided
        if self.args.category_names:
            class_names = load_class_names(self.args.category_names)
            class_labels = [class_names.get(str(cls), str(cls)) for cls in classes]
            print('Classes:', class_labels)
        else:
            print('Classes:', classes)

if __name__ == '__main__':
    predictor = Predictor()
    predictor.run()