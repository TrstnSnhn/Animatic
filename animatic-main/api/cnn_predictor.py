"""
CNN KEYPOINT PREDICTOR
======================
This module loads the trained CNN model and provides keypoint prediction
for the Animatic 2D to 3D rigging pipeline.

Replaces MediaPipe with your custom-trained CNN model.
"""

import numpy as np
import cv2
from pathlib import Path

# TensorFlow import with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not available. CNN predictor will not work.")


class CNNKeypointPredictor:
    """
    CNN-based keypoint predictor for 2D character images.
    
    This class loads a trained Keras model and predicts 21 body keypoints
    from input images.
    """
    
    # Keypoint names in order (must match training)
    KEYPOINT_NAMES = [
        'head_top', 'head_center', 'neck', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'chest_center',
        'waist_center', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_hand', 'right_hand', 'left_foot', 'right_foot'
    ]
    
    # Mapping from our keypoint names to the anime_keypoints format used in main.py
    KEYPOINT_INDEX_MAP = {
        0: 'head_top',
        1: 'head_center', 
        2: 'neck',
        3: 'left_shoulder',
        4: 'right_shoulder',
        5: 'left_elbow',
        6: 'right_elbow',
        7: 'left_wrist',
        8: 'right_wrist',
        9: 'chest_center',
        10: 'waist_center',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle',
        17: 'left_hand',
        18: 'right_hand',
        19: 'left_foot',
        20: 'right_foot'
    }
    
    def __init__(self, model_path="trained_model/best_model.keras"):
        """
        Initialize the CNN predictor.
        
        Args:
            model_path: Path to the trained Keras model file
        """
        self.model_path = Path(model_path)
        self.model = None
        self.input_size = (256, 256)  # Must match training configuration
        self.num_keypoints = 21
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained Keras model."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not installed. Cannot load CNN model.")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading CNN model from: {self.model_path}")
        self.model = keras.models.load_model(str(self.model_path))
        print("CNN model loaded successfully!")
        
        # Warm up the model with a dummy prediction
        dummy_input = np.zeros((1, *self.input_size, 3), dtype=np.float32)
        _ = self.model.predict(dummy_input, verbose=0)
        print("Model warm-up complete.")
    
    def preprocess_image(self, image):
        """
        Preprocess image for CNN input.
        
        Args:
            image: Input image (BGR format from cv2.imread or RGB)
            
        Returns:
            Preprocessed image array and original dimensions
        """
        if image is None:
            raise ValueError("Input image is None")
        
        # Store original dimensions
        original_h, original_w = image.shape[:2]
        
        # Convert BGR to RGB if needed (cv2 loads as BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR from OpenCV, convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize to model input size
        image_resized = cv2.resize(image_rgb, self.input_size)
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch, (original_w, original_h)
    
    def predict(self, image):
        """
        Predict keypoints for an image.
        
        Args:
            image: Input image (BGR or RGB format)
            
        Returns:
            numpy array of shape (21, 3) with [x, y, confidence] for each keypoint
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Preprocess
        image_batch, original_size = self.preprocess_image(image)
        original_w, original_h = original_size
        
        # Predict
        predictions = self.model.predict(image_batch, verbose=0)
        
        # Reshape from (1, 42) to (21, 2)
        keypoints_normalized = predictions[0].reshape(self.num_keypoints, 2)
        
        # Convert normalized coordinates back to pixel coordinates
        keypoints = np.zeros((self.num_keypoints, 3))
        for i in range(self.num_keypoints):
            x_norm, y_norm = keypoints_normalized[i]
            
            # Scale to original image dimensions
            x_pixel = x_norm * original_w
            y_pixel = y_norm * original_h
            
            # CNN doesn't output confidence, so we set a default high confidence
            # You could calculate confidence based on how close predictions are to image bounds
            confidence = 0.9  # Default confidence
            
            # Reduce confidence if prediction is near edges (likely less accurate)
            if x_norm < 0.05 or x_norm > 0.95 or y_norm < 0.05 or y_norm > 0.95:
                confidence = 0.6
            
            keypoints[i] = [x_pixel, y_pixel, confidence]
        
        return keypoints
    
    def predict_as_dict(self, image):
        """
        Predict keypoints and return as dictionary.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary mapping keypoint names to (x, y, confidence) tuples
        """
        keypoints_array = self.predict(image)
        
        result = {}
        for i, name in enumerate(self.KEYPOINT_NAMES):
            result[name] = {
                'x': float(keypoints_array[i, 0]),
                'y': float(keypoints_array[i, 1]),
                'confidence': float(keypoints_array[i, 2])
            }
        
        return result
    
    def get_keypoints_for_rigging(self, image):
        """
        Get keypoints in the format expected by the rigging pipeline.
        
        This method returns keypoints in the same format as the original
        MediaPipe-based processor, ensuring compatibility with existing code.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            numpy array of shape (21, 3) with [x, y, confidence] for each keypoint
        """
        return self.predict(image)


class HybridKeypointPredictor:
    """
    Hybrid predictor that uses CNN as primary and MediaPipe as fallback.
    
    This ensures robustness - if CNN fails or produces low-confidence results,
    MediaPipe can be used as a backup.
    """
    
    def __init__(self, cnn_model_path="trained_model/best_model.keras", use_mediapipe_fallback=True):
        """
        Initialize hybrid predictor.
        
        Args:
            cnn_model_path: Path to trained CNN model
            use_mediapipe_fallback: Whether to use MediaPipe as fallback
        """
        self.cnn_predictor = None
        self.mediapipe_available = False
        self.use_fallback = use_mediapipe_fallback
        
        # Try to load CNN model
        try:
            self.cnn_predictor = CNNKeypointPredictor(cnn_model_path)
            print("CNN predictor loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load CNN model: {e}")
        
        # Try to import MediaPipe as fallback
        if use_mediapipe_fallback:
            try:
                import mediapipe as mp
                self.mp_pose = mp.solutions.pose
                self.pose_detector = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    min_detection_confidence=0.5
                )
                self.mediapipe_available = True
                print("MediaPipe fallback available")
            except ImportError:
                print("MediaPipe not available for fallback")
    
    def predict(self, image, confidence_threshold=0.5):
        """
        Predict keypoints using CNN, with MediaPipe fallback if needed.
        
        Args:
            image: Input image
            confidence_threshold: Minimum average confidence to accept CNN result
            
        Returns:
            tuple: (keypoints array, method used: 'cnn' or 'mediapipe')
        """
        # Try CNN first
        if self.cnn_predictor is not None:
            try:
                keypoints = self.cnn_predictor.predict(image)
                avg_confidence = np.mean(keypoints[:, 2])
                
                if avg_confidence >= confidence_threshold:
                    return keypoints, 'cnn'
                else:
                    print(f"CNN confidence too low ({avg_confidence:.2f}), trying fallback...")
            except Exception as e:
                print(f"CNN prediction failed: {e}")
        
        # Fallback to MediaPipe
        if self.mediapipe_available and self.use_fallback:
            try:
                keypoints = self._mediapipe_predict(image)
                return keypoints, 'mediapipe'
            except Exception as e:
                print(f"MediaPipe fallback also failed: {e}")
        
        raise RuntimeError("All prediction methods failed")
    
    def _mediapipe_predict(self, image):
        """Use MediaPipe for prediction (fallback method)."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        results = self.pose_detector.process(image_rgb)
        
        if not results.pose_landmarks:
            raise ValueError("MediaPipe could not detect pose")
        
        # Convert MediaPipe landmarks to our format
        # This is a simplified conversion - you may need to adjust based on your needs
        landmarks = results.pose_landmarks.landmark
        
        keypoints = np.zeros((21, 3))
        
        # Map MediaPipe landmarks to our 21 keypoints
        mp_to_ours = {
            0: 0,   # nose -> head_center (approximate)
            11: 3,  # left_shoulder
            12: 4,  # right_shoulder
            13: 5,  # left_elbow
            14: 6,  # right_elbow
            15: 7,  # left_wrist
            16: 8,  # right_wrist
            23: 11, # left_hip
            24: 12, # right_hip
            25: 13, # left_knee
            26: 14, # right_knee
            27: 15, # left_ankle
            28: 16, # right_ankle
            19: 17, # left_index -> left_hand
            20: 18, # right_index -> right_hand
            31: 19, # left_foot_index -> left_foot
            32: 20, # right_foot_index -> right_foot
        }
        
        for mp_idx, our_idx in mp_to_ours.items():
            lm = landmarks[mp_idx]
            keypoints[our_idx] = [lm.x * width, lm.y * height, lm.visibility]
        
        # Calculate derived keypoints
        # head_top (0) - estimate above nose
        nose = landmarks[0]
        keypoints[0] = [nose.x * width, (nose.y - 0.1) * height, nose.visibility]
        
        # head_center (1) - use nose
        keypoints[1] = [nose.x * width, nose.y * height, nose.visibility]
        
        # neck (2) - midpoint of shoulders
        ls, rs = landmarks[11], landmarks[12]
        keypoints[2] = [(ls.x + rs.x) / 2 * width, (ls.y + rs.y) / 2 * height, (ls.visibility + rs.visibility) / 2]
        
        # chest_center (9) - between neck and hips
        lh, rh = landmarks[23], landmarks[24]
        neck_y = (ls.y + rs.y) / 2
        hip_y = (lh.y + rh.y) / 2
        keypoints[9] = [(ls.x + rs.x) / 2 * width, (neck_y + hip_y) / 2 * height, 0.8]
        
        # waist_center (10) - midpoint of hips
        keypoints[10] = [(lh.x + rh.x) / 2 * width, (lh.y + rh.y) / 2 * height, (lh.visibility + rh.visibility) / 2]
        
        return keypoints


# Convenience function for easy integration
def load_predictor(model_path="trained_model/best_model.keras", use_fallback=True):
    """
    Load the keypoint predictor.
    
    Args:
        model_path: Path to trained CNN model
        use_fallback: Whether to use MediaPipe as fallback
        
    Returns:
        HybridKeypointPredictor instance
    """
    return HybridKeypointPredictor(model_path, use_fallback)


# Test function
if __name__ == "__main__":
    import sys
    
    print("Testing CNN Keypoint Predictor")
    print("=" * 50)
    
    # Test with a sample image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python cnn_predictor.py <image_path>")
        print("No image provided, running basic load test...")
        
        try:
            predictor = CNNKeypointPredictor()
            print("✓ Model loaded successfully")
        except FileNotFoundError:
            print("✗ Model file not found. Make sure 'trained_model/best_model.keras' exists.")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        sys.exit(0)
    
    # Load and test with provided image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        sys.exit(1)
    
    try:
        predictor = CNNKeypointPredictor()
        keypoints = predictor.predict(image)
        
        print(f"\nPredicted keypoints for: {image_path}")
        print("-" * 50)
        for i, name in enumerate(CNNKeypointPredictor.KEYPOINT_NAMES):
            x, y, conf = keypoints[i]
            print(f"{name:20s}: ({x:7.2f}, {y:7.2f}) conf={conf:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")