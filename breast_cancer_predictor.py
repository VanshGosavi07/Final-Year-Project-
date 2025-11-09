import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import logging

logger = logging.getLogger(__name__)

class BreastCancerPredictor:
    def __init__(self, model_path=r"Modal\Breast Cancer\breast_cancer.keras"):
        """Initialize the breast cancer predictor with the trained model."""
        # Normalize path for cross-platform compatibility (Windows/Linux)
        self.model_path = model_path.replace('\\', '/')
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained Keras model."""
        try:
            # Check if file exists and is not empty
            if not os.path.exists(self.model_path):
                logger.error(f"Breast Cancer model file not found: {self.model_path}")
                logger.error(f"Current working directory: {os.getcwd()}")
                logger.error(f"Absolute path attempted: {os.path.abspath(self.model_path)}")
                return False
            
            # Check if file is not empty
            file_size = os.path.getsize(self.model_path)
            if file_size == 0:
                logger.error(f"Breast Cancer model file is empty (0 bytes): {self.model_path}")
                return False
            
            logger.info(f"Loading Breast Cancer model from: {self.model_path} ({file_size / (1024*1024):.1f} MB)")
            
            # Load model with compile=False for faster loading
            self.model = load_model(self.model_path, compile=False)
            logger.info(f"Breast Cancer model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Breast Cancer model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def preprocess_image(self, image_path):
        """Preprocess the input image for prediction."""
        try:
            # Read image using cv2
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Resize to model input size (224x224 for breast cancer model)
            img_resized = cv2.resize(img_cv, (224, 224))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def predict_image(self, image_path):
        """
        Predict the class of a breast X-ray image.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            tuple: (label, confidence) - ("Cancer: Yes (Malignant)" or "Cancer: No (Non-Malignant)", confidence_score)
        """
        if self.model is None:
            logger.warning("Breast Cancer model not loaded")
            return "Cancer: Unknown (Model Unavailable)", 0.0
        
        processed_img = self.preprocess_image(image_path)
        if processed_img is None:
            return "Cancer: Unknown (Image Processing Error)", 0.0
        
        try:
            prediction = self.model.predict(processed_img, verbose=0)
            confidence = float(prediction[0][0])
            
            if confidence >= 0.5:
                label = "Cancer: Yes (Malignant)"
            else:
                label = "Cancer: No (Non-Malignant)"
                
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
            elif "Non-Malignant" in label:
                color = (0, 255, 0)  # Green for non-malignant
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
