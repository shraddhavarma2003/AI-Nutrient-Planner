"""
Quick diagnostic script to test YOLO model loading and configuration.
"""
import os
import sys

def test_yolo_model():
    print("=" * 60)
    print("YOLO Model Diagnostic Test")
    print("=" * 60)
    
    # Check if model file exists
    model_path = "models/best.pt"
    print(f"\n1. Checking model file: {model_path}")
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"   ✓ Model file found ({size_mb:.1f} MB)")
    else:
        print(f"   ✗ Model file NOT found!")
        return
    
    # Try to load the model
    print("\n2. Loading YOLO model...")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print(f"   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return
    
    # Print model info
    print("\n3. Model Information:")
    print(f"   Task: {model.task}")
    print(f"   Number of classes: {len(model.names)}")
    print(f"   Classes: {list(model.names.values())[:10]}...")  # First 10 classes
    
    # Check if model is a detection model
    print("\n4. Model Type Check:")
    if model.task == "detect":
        print("   ✓ Model is configured for DETECTION")
    elif model.task == "classify":
        print("   ⚠ Model is configured for CLASSIFICATION (not detection)")
        print("   This may affect how predictions work!")
    else:
        print(f"   Model task type: {model.task}")
    
    # Test prediction on a sample image (if available)
    print("\n5. Testing prediction...")
    test_images = [
        "uploads/test.jpg",
        "static/sample_food.jpg",
    ]
    
    # Find first existing test image or create a dummy one
    test_image = None
    for img in test_images:
        if os.path.exists(img):
            test_image = img
            break
    
    if not test_image:
        # List any images in uploads folder
        if os.path.exists("uploads"):
            files = os.listdir("uploads")
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                test_image = os.path.join("uploads", image_files[0])
    
    if test_image:
        print(f"   Testing with: {test_image}")
        try:
            results = model(test_image, verbose=False)
            
            for result in results:
                if model.task == "classify":
                    # Classification model output
                    probs = result.probs
                    if probs is not None:
                        top5_indices = probs.top5
                        top5_confs = probs.top5conf.tolist()
                        print(f"\n   Top 5 predictions (classification model):")
                        for idx, conf in zip(top5_indices, top5_confs):
                            class_name = model.names[idx]
                            print(f"     - {class_name}: {conf:.1%}")
                else:
                    # Detection model output
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        print(f"\n   Detections found: {len(boxes)}")
                        for i, box in enumerate(boxes):
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            class_name = model.names[cls]
                            print(f"     {i+1}. {class_name}: {conf:.1%}")
                    else:
                        print("   ✗ No detections found in test image")
                        print("   Possible reasons:")
                        print("     - Confidence threshold too high")
                        print("     - Model not trained on this type of food")
                        print("     - Image quality/format issues")
        except Exception as e:
            print(f"   ✗ Prediction failed: {e}")
    else:
        print("   No test image available. Upload an image to test.")
    
    # Print summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    if model.task == "classify":
        print("⚠ ISSUE FOUND: The model is a CLASSIFICATION model, not DETECTION")
        print("  Classification models use result.probs, not result.boxes")
        print("  The yolo_service.py needs to be updated to handle classification models.")
    else:
        print("Model appears to be configured correctly for detection.")
        print("If no food is detected, check:")
        print("  1. Image quality and lighting")
        print("  2. Whether the food type is in the model's training classes")
        print("  3. Confidence threshold (currently 0.25)")

if __name__ == "__main__":
    test_yolo_model()
