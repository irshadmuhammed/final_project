import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm
from typing import Optional, Tuple, Dict, Any, Union

class GradCAMExplainer:
    """
    A production-ready Grad-CAM and Grad-CAM++ explainer for Medical Image Classification models.
    Supports single-channel (grayscale) and 3-channel (RGB) images.
    Compatibile with TensorFlow/Keras models.
    """

    def __init__(self, model: tf.keras.Model, target_layer_name: Optional[str] = None):
        """
        Initialize the GradCAMExplainer.

        Args:
            model (tf.keras.Model): The trained Keras model.
            target_layer_name (str, optional): Name of the last convolutional layer.
                                               If None, it's automatically detected.
        """
        self.model = model
        self.target_layer_name = target_layer_name
        
        if self.target_layer_name is None:
            self.target_layer_name = self._find_target_layer()
            print(f"DEBUG: Automatically detected target layer: {self.target_layer_name}")

        self.grad_model = tf.keras.models.Model(
            [self.model.inputs], 
            [self.model.get_layer(self.target_layer_name).output, self.model.output]
        )

    def _find_target_layer(self) -> str:
        """Finds the last convolutional layer in the model."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Conv3D)):
                return layer.name
            # Handle nested models (e.g. ResNet50 as a layer)
            if hasattr(layer, 'layers'):
                for sub_layer in reversed(layer.layers):
                     if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.Conv3D)):
                        return sub_layer.name
        raise ValueError("Could not find a convolutional layer in the model.")

    def preprocess_image(self, img_path: str, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Preprocesses a medical image for the model.

        Args:
            img_path (str): Path to the image file.
            target_size (tuple): (height, width) for resizing.

        Returns:
            np.ndarray: Preprocessed image batch of shape (1, H, W, C).
        """
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found at {img_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            
            # Normalize to [0, 1] - Adjust if your model expects different normalization
            img_array = img.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            return img_array
        except Exception as e:
            raise RuntimeError(f"Error during preprocessing: {e}")

    def compute_heatmap(self, image: np.ndarray, class_index: int, method: str = "gradcam") -> np.ndarray:
        """
        Computes the Grad-CAM or Grad-CAM++ heatmap.

        Args:
            image (np.ndarray): Input image batch (1, H, W, C).
            class_index (int): Index of the target class.
            method (str): 'gradcam' or 'gradcam++'.

        Returns:
            np.ndarray: The raw heatmap (H, W).
        """
        # Ensure image is a tensor for GradientTape
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(image)  # Use watch since models usually pass non-variable tensors
            conv_outputs, predictions = self.grad_model(image)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        
        if method == "gradcam++":
            heatmap = self._compute_gradcam_plusplus(conv_outputs, grads, loss)
        else:
            heatmap = self._compute_gradcam(conv_outputs, grads)

        # Normalize the heatmap between 0 and 1
        heatmap = np.maximum(heatmap, 0)
        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat

        return heatmap

    def _compute_gradcam(self, conv_outputs, grads) -> np.ndarray:
        """Standard Grad-CAM logic."""
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        
        # Weighted sum
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        return heatmap.numpy()

    def _compute_gradcam_plusplus(self, conv_outputs, grads, score) -> np.ndarray:
        """Grad-CAM++ logic for better localization."""
        conv_outputs = conv_outputs[0]
        grads = grads[0]
        score = score[0]

        # First derivative
        first_derivative = tf.exp(score) * grads
        second_derivative = tf.exp(score) * grads * grads
        third_derivative = tf.exp(score) * grads * grads * grads

        global_sum = tf.reduce_sum(conv_outputs, axis=(0, 1))
        
        alpha_denom = 2.0 * second_derivative + third_derivative * global_sum[tf.newaxis, tf.newaxis, :]
        alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
        
        alphas = second_derivative / alpha_denom
        
        weights = tf.maximum(first_derivative, 0.0)
        alphas_thresholding = tf.where(weights != 0, alphas, tf.zeros_like(alphas))

        alpha_normalization_constant = tf.reduce_sum(alphas_thresholding, axis=(0,1))
        alpha_normalization_constant = tf.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, tf.ones_like(alpha_normalization_constant))
        
        alphas /= alpha_normalization_constant[tf.newaxis, tf.newaxis, :]

        deep_linearization_weights = tf.reduce_sum(weights * alphas, axis=(0,1))
        
        heatmap = tf.reduce_sum(deep_linearization_weights * conv_outputs, axis=-1)
        return heatmap.numpy()

    def overlay_heatmap(self, heatmap: np.ndarray, original_image: np.ndarray, alpha: float = 0.4, colormap=cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlays the heatmap on the original image.

        Args:
            heatmap (np.ndarray): The raw heatmap (H, W).
            original_image (np.ndarray): The original image (H, W, 3) or (H, W).
            alpha (float): Transparency factor for the heatmap overlay.
            colormap: OpenCV colormap to apply.

        Returns:
            np.ndarray: The superimposed image (H, W, 3), range [0, 255], uint8.
        """
        # Resize heatmap to match image dimensions
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Rescale heatmap to 0-255
        heatmap = np.uint8(255 * heatmap)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, colormap)
        
        # Ensure original_image is uint8 [0, 255]
        if original_image.dtype != np.uint8:
             original_image = np.uint8(255 * original_image) if np.max(original_image) <= 1.0 else np.uint8(original_image)
        
        # Convert grayscale to RGB if necessary for blending
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif original_image.shape[2] == 1: # Single channel with dim
             original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

        # Superimpose
        superimposed_img = heatmap_colored * alpha + original_image * (1 - alpha)
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        return superimposed_img

    def get_bounding_box(self, heatmap: np.ndarray, original_image_shape: Tuple[int, int], threshold: float = 0.5) -> Optional[Tuple[int, int, int, int]]:
        """
        Extracts a bounding box from the heatmap using thresholding.

        Args:
            heatmap (np.ndarray): Raw heatmap.
            original_image_shape (tuple): (H, W) of original image.
            threshold (float): Percentage of max intensity to threshold.

        Returns:
            tuple: (x, y, w, h) bounding box or None if no contours found.
        """
        heatmap_resized = cv2.resize(heatmap, (original_image_shape[1], original_image_shape[0]))
        heatmap_norm = np.uint8(255 * heatmap_resized)
        
        # Threshold
        _, thresh = cv2.threshold(heatmap_norm, int(255 * threshold), 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            return (x, y, w, h)
        return None

    def explain(self, 
                image_input: Union[str, np.ndarray], 
                class_index: Optional[int] = None, 
                method: str = "gradcam",
                use_gradcam_plusplus: bool = False) -> Dict[str, Any]:
        """
        Main method to generate explanations.

        Args:
            image_input (str or np.ndarray): Path to image or preprocessed numpy array.
            class_index (int, optional): Target class index. if None, uses predicted class.
            method (str): 'gradcam' (default). 'gradcam++' is also supported directly or via next arg.
            use_gradcam_plusplus (bool): Flag to force Grad-CAM++.

        Returns:
            dict: {
                "heatmap": np.ndarray (H, W),
                "superimposed_image": np.ndarray (H, W, 3),
                "prediction_confidence": float,
                "predicted_class": int,
                "bounding_box": tuple(x, y, w, h) or None
            }
        """
        # Determine internal method
        if use_gradcam_plusplus:
            method = "gradcam++"

        # Handle Input
        if isinstance(image_input, str):
            # Infer target size from model input shape
            input_shape = self.model.input_shape[1:3] 
            processed_image = self.preprocess_image(image_input, input_shape)
            original_image_path = image_input
            # Load original for overlay
            original_bgr = cv2.imread(original_image_path)
            original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        else:
            processed_image = image_input
            # Cannot infer original image perfectly if only array is passed, assume normalized 0-1
            original_rgb = np.squeeze(processed_image)

        # Get Predictions
        preds = self.model.predict(processed_image)
        if class_index is None:
            class_index = np.argmax(preds[0])
        
        confidence = float(preds[0][class_index])

        # Compute Heatmap
        heatmap = self.compute_heatmap(processed_image, class_index, method=method)

        # Overlay
        superimposed = self.overlay_heatmap(heatmap, original_rgb)

        # Bounding Box (ROI)
        bbox = self.get_bounding_box(heatmap, original_rgb.shape[:2])

        return {
            "heatmap": heatmap,
            "superimposed_image": superimposed,
            "prediction_confidence": confidence,
            "predicted_class": int(class_index),
            "bounding_box": bbox
        }

# --- Example Usage ---
if __name__ == "__main__":
    # Mock Model for demonstration purposes
    # Ideally, load your trained .h5 model here
    try:
        model = tf.keras.applications.DenseNet121(weights='imagenet')
        explainer = GradCAMExplainer(model)
        
        # Use a dummy image or prompt user
        print("Model loaded successfully. Instantiate GradCAMExplainer with your model and use .explain() method.")
        
        # Example call (Commented out as we need a real image)
        # result = explainer.explain("test_chest_xray.jpg", class_index=0)
        # cv2.imwrite("gradcam_result.jpg", cv2.cvtColor(result["superimposed_image"], cv2.COLOR_RGB2BGR))
        
    except Exception as e:
        print(f"Setup failed (Example usage requires internet for DenseNet weights): {e}")
