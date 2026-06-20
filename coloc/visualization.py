import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def generate_smart_color_palette(kmeans):
    """
    Treats the K cluster centroids in the PCA space as a new dataset
    and reduces their dimensions from D-dim to 3D. Scales these 3D
    coordinates to [0, 1] to get distinct, perceptually aligned RGB colors.
    """
    centroids_pca = kmeans.cluster_centers_
    
    # Project centroids to 3D
    pca_color = PCA(n_components=3)
    centroids_3d = pca_color.fit_transform(centroids_pca)
    
    # Scale to [0.0, 1.0] range
    scaler = MinMaxScaler()
    color_map = scaler.fit_transform(centroids_3d)
    
    return color_map

def create_cluster_map(frame, patch_labels, color_map, patch_grid):
    """
    Creates a full-resolution blended image showing all K-Means clusters simultaneously.
    Uses the provided color_map to map cluster IDs to RGB colors.
    """
    h, w, _ = frame.shape
    
    # Create low-resolution mask grid
    mask = patch_labels.reshape(patch_grid)
    mask_rgb_lowres = np.zeros((*patch_grid, 3), dtype=np.float32)
    
    # Color the low-res mask
    for k in range(len(color_map)):
        mask_rgb_lowres[mask == k] = color_map[k]
        
    # Upscale the low-res mask using nearest-neighbor interpolation to preserve boundaries
    mask_rgb_full = cv2.resize(mask_rgb_lowres, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Normalize original frame to float and blend
    frame_float = frame.astype(np.float32) / 255.0
    blended_img = cv2.addWeighted(frame_float, 0.6, mask_rgb_full, 0.4, 0)
    
    return (blended_img * 255).astype(np.uint8)

def create_binary_overlay(frame, patch_labels, target_cluster, patch_grid, color=(0, 0, 150), alpha=0.7):
    """
    Creates a full-resolution blended image showing only the target object cluster
    as a semi-transparent colored overlay (default is red).
    """
    h, w, _ = frame.shape
    
    # Create binary mask grid
    mask = (patch_labels.reshape(patch_grid) == target_cluster).astype(np.uint8)
    
    # Upscale binary mask to full frame size
    mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create colored overlay image
    overlay = np.zeros_like(frame, dtype=np.uint8)
    # color parameter is RGB, convert to BGR for standard OpenCV usage if needed,
    # but let's treat the inputs as BGR since cv2 reads/writes BGR by default.
    overlay[...] = color
    
    # Apply overlay where mask is active
    masked_img = cv2.bitwise_and(overlay, overlay, mask=mask_full)
    
    # Blend with original
    blended_img = cv2.addWeighted(frame, 1.0, masked_img, alpha, 0)
    
    return blended_img

def save_video_with_overlays(input_video_path, output_video_path, encoder, pca, kmeans, color_map, object_cluster=None, progress_callback=None):
    """
    Processes an input MP4 video frame-by-frame, extracts DINOv2 features, projects them,
    predicts cluster IDs, overlays them, and writes the output to a new MP4 video.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use MP4V codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Convert BGR color map to RGB for display blending
    # (cv2 reads frame as BGR, so if we use color_map as RGB, we want to blend correctly)
    color_map_bgr = color_map[..., ::-1]  # BGR representation of colors
    
    try:
        for i in tqdm(range(frame_count), desc="Processing video frames"):
            ret, frame = cap.read()
            if not ret:
                break
                
            # 1. Convert frame to RGB for encoder extraction
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. Extract features
            features = encoder.extract_features([frame_rgb], batch_size=1, show_progress=False)
            
            # 3. PCA + K-Means prediction
            features_pca = pca.transform(features)
            patch_labels = kmeans.predict(features_pca)
            
            # 4. Generate visual overlay
            if object_cluster is not None:
                # Highlight only the target object cluster (e.g., in red)
                highlighted_frame = create_binary_overlay(
                    frame,
                    patch_labels,
                    object_cluster,
                    encoder.patch_grid,
                    color=(0, 0, 180),  # Red in BGR
                    alpha=0.6
                )
            else:
                # Display all clusters with the perceptually mapped color palette
                highlighted_frame = create_cluster_map(
                    frame,
                    patch_labels,
                    color_map_bgr,
                    encoder.patch_grid
                )
                
            out.write(highlighted_frame)
            
            if progress_callback is not None:
                progress_callback((i + 1) / frame_count)
    finally:
        cap.release()
        out.release()
        
    print(f"Output video successfully saved to: {output_video_path}")
    return output_video_path
