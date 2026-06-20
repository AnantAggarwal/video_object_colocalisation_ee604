import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPVisionModel

class BaseEncoder:
    """Base interface for feature extraction models."""
    def extract_features(self, frames, batch_size=32, show_progress=True):
        raise NotImplementedError("Subclasses must implement extract_features method.")

class DINOv2Encoder(BaseEncoder):
    """Frozen DINOv2 feature extractor for dense patch-embeddings."""
    def __init__(self, model_name="facebook/dinov2-small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DINOv2] Using device: {self.device}")
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        
        self.patch_size = self.model.config.patch_size
        self.target_size = 224  # Standard training size
        self.grid_size = self.target_size // self.patch_size
        self.patch_grid = (self.grid_size, self.grid_size)
        self.patches_per_frame = self.grid_size * self.grid_size
        self.feature_dim = self.model.config.hidden_size

    @torch.no_grad()
    def extract_features(self, frames, batch_size=32, show_progress=True):
        if not frames:
            return np.empty((0, self.feature_dim), dtype=np.float32)
            
        all_features = []
        iterator = range(0, len(frames), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting DINOv2 features")
            
        for i in iterator:
            batch_frames = frames[i:i + batch_size]
            inputs = self.processor(
                images=batch_frames,
                return_tensors="pt",
                do_resize=True,
                size={"height": self.target_size, "width": self.target_size}
            ).to(self.device)
            
            outputs = self.model(**inputs)
            # Slice off the CLS token (index 0) to get patch-level tokens only
            patch_features = outputs.last_hidden_state[:, 1:, :]
            num_frames, num_patches, dim = patch_features.shape
            flat_features = patch_features.reshape(num_frames * num_patches, dim).cpu().numpy()
            all_features.append(flat_features)
            
        return np.concatenate(all_features, axis=0)

class SAMEncoder(BaseEncoder):
    """Segment Anything Model (SAM) image encoder for dense feature extraction."""
    def __init__(self, checkpoint_path, model_type="vit_b", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SAM] Using device: {self.device}")
        
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment-anything is not installed. "
                "Install it using: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
            
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM checkpoint not found at: {checkpoint_path}")
            
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        
        # SAM ViT-B image encoder produces 64x64 feature map (4096 patches) with 256 channels
        self.grid_size = 64
        self.patch_grid = (self.grid_size, self.grid_size)
        self.patches_per_frame = self.grid_size * self.grid_size
        self.feature_dim = 256

    @torch.no_grad()
    def extract_features(self, frames, batch_size=1, show_progress=True):
        """SAM predictor extracts image embeddings frame-by-frame."""
        if not frames:
            return np.empty((0, self.feature_dim), dtype=np.float32)
            
        all_features = []
        iterator = frames
        if show_progress:
            iterator = tqdm(frames, desc="Extracting SAM features")
            
        for frame in iterator:
            self.predictor.set_image(frame)
            # Embeddings shape: (1, 256, 64, 64)
            features = self.predictor.get_image_embedding().cpu().numpy()
            c, h, w = features.shape[1], features.shape[2], features.shape[3]
            # Reshape to (4096, 256)
            flat_features = features.reshape(c, h * w).transpose(1, 0)
            all_features.append(flat_features)
            
        return np.vstack(all_features)

class CLIPEncoder(BaseEncoder):
    """Contrastive Language-Image Pre-training (CLIP) vision encoder for patch extraction."""
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CLIP] Using device: {self.device}")
        
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name).to(self.device).eval()
        
        # Determine patch size from model config (typically 32 or 16)
        self.patch_size = self.model.config.patch_size
        self.target_size = self.model.config.image_size
        self.grid_size = self.target_size // self.patch_size
        self.patch_grid = (self.grid_size, self.grid_size)
        self.patches_per_frame = self.grid_size * self.grid_size
        self.feature_dim = self.model.config.hidden_size

    @torch.no_grad()
    def extract_features(self, frames, batch_size=32, show_progress=True):
        if not frames:
            return np.empty((0, self.feature_dim), dtype=np.float32)
            
        all_features = []
        iterator = range(0, len(frames), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting CLIP features")
            
        for i in iterator:
            batch_frames = frames[i:i + batch_size]
            inputs = self.processor(images=batch_frames, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # Slice off the CLS token (index 0)
            patch_features = outputs.last_hidden_state[:, 1:, :]
            num_frames, num_patches, dim = patch_features.shape
            flat_features = patch_features.reshape(num_frames * num_patches, dim).cpu().numpy()
            all_features.append(flat_features)
            
        return np.concatenate(all_features, axis=0)
