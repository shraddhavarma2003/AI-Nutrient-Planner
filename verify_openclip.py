import sys
import os
from pathlib import Path
from PIL import Image
import torch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from services.continental_retrieval import get_continental_retrieval_system
    
    print("--- Starting OpenCLIP Verification ---")
    system = get_continental_retrieval_system()
    
    print(f"System initialized on: {system.device}")
    print(f"Text features shape: {system.text_features.shape}")
    
    # Test image encoding with a blank image
    dummy_img = Image.new('RGB', (224, 224), color='white')
    img_embed = system.encode_image(dummy_img)
    
    print(f"Image embedding shape: {img_embed.shape}")
    print(f"Image embedding norm: {torch.norm(img_embed).item():.4f}")
    
    # Test retrieval
    results = system.retrieve_top_k(img_embed, k=3)
    print("\nTop 3 Predictions for blank image:")
    for p in results['top_k_predictions']:
        print(f"- {p['dish']}: {p['score']}")
        
    print("\n--- Verification Complete ---")

except Exception as e:
    print(f"CRITICAL ERROR during verification: {e}")
    import traceback
    traceback.print_exc()
