import os
import torch
import logging
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import CLIPProcessor, CLIPModel

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContinentalRetrieval")

class ContinentalRetrievalSystem:
    """
    A production-grade CLIP-based retrieval system for continental dishes.
    
    Architecture:
    1. Eager loading of CLIP model.
    2. One-time precomputation of dish text embeddings using multiple prompts.
    3. Runtime image encoding and cosine similarity search.
    """
    
    MODEL_ID = "openai/clip-vit-base-patch32"
    DISHES_PATH = Path(__file__).parent.parent.parent / "data" / "dishes_continental.txt"
    CONFIDENCE_THRESHOLD = 0.20
    
    def __init__(self):
        # State variables
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = None
        self.processor = None
        self.dish_names: List[str] = []
        self.text_features: Optional[torch.Tensor] = None
        
        logger.info(f"Initializing ContinentalRetrievalSystem on {self.device} ({self.dtype})")
        
        # OFFLINE / STARTUP STAGES
        self.load_model()
        self.load_dishes_txt()
        self.build_text_index()

    def load_model(self):
        """A. Loads CLIP model and processor ONCE."""
        logger.info(f"Loading weights for {self.MODEL_ID}...")
        self.model = CLIPModel.from_pretrained(self.MODEL_ID).to(self.device).to(self.dtype)
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
        self.model.eval()
        
        # Sanity check logging
        logger.info(f"Model loaded. Precision: {self.model.dtype}")

    def load_dishes_txt(self):
        """B. Reads dish names from the provided TXT file."""
        if not self.DISHES_PATH.exists():
            raise FileNotFoundError(f"Continental dishes file not found at {self.DISHES_PATH}")
            
        with open(self.DISHES_PATH, "r", encoding="utf-8") as f:
            self.dish_names = [line.strip() for line in f if line.strip()]
            
        logger.info(f"Loaded {len(self.dish_names)} continental dishes.")

    @torch.inference_mode()
    def build_text_index(self):
        """C. Generates and caches text embeddings using prompt ensemble."""
        cache_path = self.DISHES_PATH.parent / "continental_embeddings.pt"
        
        # Try loading from cache first
        if cache_path.exists():
            try:
                logger.info(f"Loading cached text index from {cache_path}...")
                cached_data = torch.load(cache_path, map_location=self.device)
                
                # Validation: Ensure cache matches current dishes
                if cached_data.get("dish_names") == self.dish_names:
                    self.text_features = cached_data["features"].to(self.device).to(self.dtype)
                    logger.info(f"✓ Cache loaded. Shape: {self.text_features.shape}")
                    return
                else:
                    logger.warning("Cache mismatch (dish list changed). Rebuilding index...")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Rebuilding...")
        
        logger.info("Building text embedding index (Offline Stage)...")
        
        # Use only the two requested prompt templates
        templates = ["a photo of {}", "a photo of {} food"]
        
        all_features = []
        
        # Batch process for efficiency
        batch_size = 32
        total_batches = (len(self.dish_names) + batch_size - 1) // batch_size
        
        for i in range(0, len(self.dish_names), batch_size):
            batch_num = (i // batch_size) + 1
            if batch_num % 5 == 0:
                logger.info(f"Processing batch {batch_num}/{total_batches}...")
                
            batch_dishes = self.dish_names[i : i + batch_size]
            
            # For each template, compute features
            template_features = []
            for template in templates:
                prompts = [template.format(dish) for dish in batch_dishes]
                
                inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                features = self.model.get_text_features(**inputs)
                
                # Normalize features
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                template_features.append(features)
            
            # Average the embeddings across the two templates (Prompt Ensembling)
            # Stack: [2, batch_size, dim] -> Mean: [batch_size, dim]
            averaged_batch = torch.stack(template_features).mean(dim=0)
            
            # Re-normalize after averaging
            averaged_batch = averaged_batch / averaged_batch.norm(p=2, dim=-1, keepdim=True)
            all_features.append(averaged_batch)
            
        # Final cached index: [num_dishes, dim]
        self.text_features = torch.cat(all_features, dim=0)
        
        # Save to cache
        try:
            torch.save({
                "dish_names": self.dish_names,
                "features": self.text_features
            }, cache_path)
            logger.info(f"✓ Index built and cached to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            
        logger.info(f"Text index ready. Shape: {self.text_features.shape}")

    @torch.inference_mode()
    def encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        """D. Encodes a single image into normalized CLIP space."""
        # Preprocessing (Fix resizing and normalization per model specs)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Move inputs to correct dtype
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        
        # Forward pass
        image_features = self.model.get_image_features(**inputs)
        
        # Sanity check logging
        # logger.debug(f"Image feature shape: {image_features.shape}")
        
        # Normalization (MANDATORY)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    def retrieve_top_k(self, image_features: torch.Tensor, k: int = 5) -> Dict[str, Any]:
        """E. Computes cosine similarity and ranks Top-K."""
        # Matrix multiplication = Cosine Similarity (since vectors are normalized)
        # [1, dim] @ [dim, num_dishes] -> [1, num_dishes]
        similarities = (image_features @ self.text_features.T).squeeze(0)
        
        # Get Top-K
        top_scores, top_indices = torch.topk(similarities, k=k)
        
        predictions = []
        for score, idx in zip(top_scores, top_indices):
            predictions.append({
                "dish": self.dish_names[idx.item()],
                "score": round(score.item(), 4)
            })
            
        max_score = predictions[0]["score"]
        status = "ok" if max_score >= self.CONFIDENCE_THRESHOLD else "unknown"
        
        return {
            "top_k_predictions": predictions,
            "confidence": max_score,
            "status": status,
            "message": None if status == "ok" else "Unknown continental dish"
        }

    def main_inference(self, image: Any, k: int = 5) -> Dict[str, Any]:
        """F. Orchestrates the full online inference flow."""
        try:
            # Handle path vs PIL
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")
            
            # Step 1: Online Image Encoding
            img_embed = self.encode_image(image)
            
            # Step 2: Cosine Similarity Matching
            result = self.retrieve_top_k(img_embed, k=k)
            return result
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                "top_k_predictions": [],
                "confidence": 0.0,
                "status": "error",
                "message": str(e)
            }

# GLOBAL SINGLETON INSTANCE
_system = None

def get_continental_retrieval_system():
    global _system
    if _system is None:
        _system = ContinentalRetrievalSystem()
    return _system

if __name__ == "__main__":
    # Quick sanity check
    sys = get_continental_retrieval_system()
    print("System initialized successfully.")
