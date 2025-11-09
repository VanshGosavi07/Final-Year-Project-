import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import logging

logger = logging.getLogger(__name__)

class LungCancerPredictor:
    def __init__(self, model_path=r"Modal\Lung Cancer\Lung Cancer.keras"):
        """Initialize the lung cancer predictor with the trained model."""
        # Normalize path for cross-platform compatibility (Windows/Linux)
        self.model_path = model_path.replace('\\', '/')
        self.model = None
        self.class_labels = ["Benign", "Malignant", "Normal"]
        self.load_model()
    
    def load_model(self):
        """Load the trained Keras model."""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                logger.info(f"Lung Cancer model loaded successfully from {self.model_path}")
                return True
            else:
                logger.error(f"Lung Cancer model file not found: {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading Lung Cancer model: {str(e)}")
            return False
    
    def preprocess_image(self, image_path):
        """Preprocess the input image for prediction."""
        try:
            # Normalize the path to handle both forward and backward slashes
            normalized_path = os.path.normpath(image_path)
            img = cv2.imread(normalized_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Resize to model input size (128x128 for lung cancer model)
            img = cv2.resize(img, (128, 128))
            img = img.astype('float32') / 255.0
            img = img.reshape(1, 128, 128, 1)
            return img
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def predict_image(self, image_path):
        """
        Predict the class of a lung X-ray image.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            tuple: (label, confidence) - (predicted_class, confidence_score)
        """
        if self.model is None:
            logger.warning("Lung Cancer model not loaded")
            return "Cancer: Unknown (Model Unavailable)", 0.0
        
        processed_img = self.preprocess_image(image_path)
        if processed_img is None:
            return "Cancer: Unknown (Image Processing Error)", 0.0
        
        try:
            prediction = self.model.predict(processed_img, verbose=0)
            predicted_class_index = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            predicted_class = self.class_labels[predicted_class_index]
            
            # Format label consistently with breast cancer predictor
            if predicted_class == "Malignant":
                label = "Cancer: Yes (Malignant)"
            elif predicted_class == "Benign":
                label = "Cancer: No (Benign)"
            elif predicted_class == "Normal":
                label = "Cancer: No (Normal)"
            else:
                label = f"Cancer: {predicted_class}"
                
            return label, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed for {image_path}: {str(e)}")
            return "Cancer: Unknown (Prediction Error)", 0.0
    
    def predict_with_visualization(self, image_path, output_path):
        """
        Predict and create visualization with bounding box and label.
        
        Args:
            image_path (str): Path to the input image
            output_path (str): Path to save the output image
            
        Returns:
            tuple: (label, confidence, output_path)
        """
        # Get prediction
        label, confidence = self.predict_image(image_path)
        
        try:
            # Load original image for visualization
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                return label, confidence, None
            
            img_copy = img_cv.copy()
            
            # Determine color based on prediction
            if "Malignant" in label:
                color = (0, 0, 255)  # Red for malignant
            elif "Benign" in label or "Normal" in label or "No" in label:
                color = (0, 255, 0)  # Green for benign/normal
            else:
                color = (255, 255, 0)  # Yellow for unknown
            
            # Add bounding box
            height, width, _ = img_copy.shape
            padding = 100
            start_point = (padding, padding)
            end_point = (width - padding, height - padding)
            img_with_rectangle = cv2.rectangle(img_copy, start_point, end_point, color, 5)
            
            # Add text label
            img_with_text = cv2.putText(img_with_rectangle, label, (50, height - 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            # Save the output image
            cv2.imwrite(output_path, img_with_text)
            
            return label, confidence, output_path
            
        except Exception as e:
            logger.error(f"Error creating visualization for {image_path}: {str(e)}")
            return label, confidence, None
