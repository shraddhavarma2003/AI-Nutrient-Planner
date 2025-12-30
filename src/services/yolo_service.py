"""
YOLO Food Recognition Service

Uses YOLOv8/v11 model (best.pt) for food detection in images.
Integrates with the nutrition pipeline to identify foods from uploaded images.
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path


class YOLOFoodRecognizer:
    """
    YOLO-based food recognition service.
    
    Uses the trained best.pt model to detect food items in images.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize YOLO food recognizer.
        
        Args:
            model_path: Path to best.pt model file
        """
        self.model = None
        self.model_path = model_path or self._find_model()
        self._initialized = False
        
        self._load_model()
    
    def _find_model(self) -> str:
        """Find the best.pt model in common locations."""
        # Check various locations
        locations = [
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "best.pt"),
            os.path.join(os.path.dirname(__file__), "..", "models", "best.pt"),
            "models/best.pt",
            "best.pt",
        ]
        
        for loc in locations:
            if os.path.exists(loc):
                return os.path.abspath(loc)
        
        return os.path.join(os.path.dirname(__file__), "..", "..", "models", "best.pt")
    
    def _load_model(self):
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO
            
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                self._initialized = True
                print(f"[YOLO] Model loaded successfully from: {self.model_path}")
                
                # Print model info
                if hasattr(self.model, 'names'):
                    print(f"[YOLO] Classes: {self.model.names}")
            else:
                print(f"[YOLO] Model file not found: {self.model_path}")
                
        except ImportError:
            print("[YOLO] ultralytics not installed. Run: pip install ultralytics")
        except Exception as e:
            print(f"[YOLO] Failed to load model: {e}")
    
    @property
    def is_available(self) -> bool:
        """Check if model is ready."""
        return self._initialized and self.model is not None
    
    def predict(self, image_path: str, confidence_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Classify or detect food items in an image.
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Dict with detected/classified food name, confidence, and all predictions
        """
        if not self.is_available:
            return {
                "success": False,
                "error": "YOLO model not available",
                "food_name": None,
                "confidence": 0.0,
                "detections": []
            }
        
        try:
            # Run inference
            print(f"[YOLO] Running prediction on: {image_path}")
            print(f"[YOLO] Model task: {self.model.task}")
            print(f"[YOLO] Model classes (count): {len(self.model.names)}")
            results = self.model(image_path, verbose=False)
            
            # Handle CLASSIFICATION models (uses result.probs)
            if self.model.task == "classify":
                for result in results:
                    probs = result.probs
                    if probs is not None:
                        # Get top prediction
                        top1_idx = probs.top1
                        top1_conf = float(probs.top1conf)
                        class_name = self.model.names[top1_idx]
                        
                        # Get top 5 for debugging
                        top5_indices = probs.top5
                        top5_confs = probs.top5conf.tolist()
                        all_predictions = [f"{self.model.names[idx]}: {conf:.1%}" 
                                          for idx, conf in zip(top5_indices, top5_confs)]
                        print(f"[YOLO] Top 5 classifications: {all_predictions}")
                        
                        if top1_conf >= confidence_threshold:
                            return {
                                "success": True,
                                "food_name": class_name,
                                "confidence": top1_conf,
                                "detections": [{"class_name": self.model.names[idx], 
                                               "confidence": conf} 
                                              for idx, conf in zip(top5_indices, top5_confs)],
                                "total_detections": 1,
                                "model_type": "classification"
                            }
                        else:
                            print(f"[YOLO] Best confidence {top1_conf:.1%} below threshold {confidence_threshold:.1%}")
                            return {
                                "success": True,
                                "food_name": None,
                                "confidence": top1_conf,
                                "detections": [],
                                "message": f"Best prediction ({class_name}) below confidence threshold"
                            }
                    else:
                        print("[YOLO] No probs in classification result")
                
                return {
                    "success": True,
                    "food_name": None,
                    "confidence": 0.0,
                    "detections": [],
                    "message": "No classification results"
                }
            
            # Handle DETECTION models (uses result.boxes)
            detections = []
            best_detection = None
            best_confidence = 0.0
            all_raw_detections = []  # For debugging
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    print(f"[YOLO] No boxes in result")
                    continue
                
                print(f"[YOLO] Found {len(boxes)} raw detections")
                
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls] if cls in self.model.names else f"class_{cls}"
                    
                    # Log ALL detections for debugging
                    all_raw_detections.append(f"{class_name}: {conf:.1%}")
                    
                    detection = {
                        "class_name": class_name,
                        "confidence": conf,
                        "box": box.xyxy[0].tolist() if hasattr(box, 'xyxy') else None
                    }
                    
                    if conf >= confidence_threshold:
                        detections.append(detection)
                        
                        if conf > best_confidence:
                            best_confidence = conf
                            best_detection = detection
            
            # Debug: show all raw detections
            if all_raw_detections:
                print(f"[YOLO] All detections (including low confidence): {all_raw_detections}")
            else:
                print(f"[YOLO] No objects detected in image")
            
            if best_detection:
                return {
                    "success": True,
                    "food_name": best_detection["class_name"],
                    "confidence": best_detection["confidence"],
                    "detections": detections,
                    "total_detections": len(detections),
                    "model_type": "detection"
                }
            else:
                return {
                    "success": True,
                    "food_name": None,
                    "confidence": 0.0,
                    "detections": [],
                    "message": "No food detected above confidence threshold"
                }
                
        except Exception as e:
            print(f"[YOLO] Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "food_name": None,
                "confidence": 0.0,
                "detections": []
            }
    
    def get_class_names(self) -> List[str]:
        """Get list of class names the model can detect."""
        if self.is_available and hasattr(self.model, 'names'):
            return list(self.model.names.values())
        return []


# Global singleton
_yolo_recognizer: Optional[YOLOFoodRecognizer] = None


def get_yolo_recognizer() -> YOLOFoodRecognizer:
    """Get or create the global YOLO recognizer instance."""
    global _yolo_recognizer
    if _yolo_recognizer is None:
        _yolo_recognizer = YOLOFoodRecognizer()
    return _yolo_recognizer


def recognize_food(image_path: str) -> Dict[str, Any]:
    """
    Convenience function to recognize food in an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dict with food_name, confidence, and detection details
    """
    recognizer = get_yolo_recognizer()
    return recognizer.predict(image_path)
