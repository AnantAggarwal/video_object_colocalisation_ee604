import os
import sys
import argparse
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Library package imports
from coloc.data import get_frame_paths, load_frames
from coloc.models import DINOv2Encoder, SAMEncoder, CLIPEncoder
from coloc.pca import EmbeddingPCA
from coloc.clustering import FeatureClustering
from coloc.evaluation import calculate_corloc
from coloc.visualization import generate_smart_color_palette, create_cluster_map

# Operations script imports
from scripts.download_data import download_and_extract
from scripts.run_demo import launch_demo

def train_pipeline(args):
    """Executes the feature extraction, PCA compression, and KMeans clustering training pipeline."""
    print("\n=== STARTING TRAINING PIPELINE ===")
    
    # 1. Discover and load dataset frames
    print("Finding frame paths...")
    frame_paths = get_frame_paths(args.data_dir, args.category, sample_rate=args.sample_rate)
    if not frame_paths:
        print(f"Error: No frame images found for category '{args.category}' under {args.data_dir}")
        print("Please run `python main.py --action download` first.")
        sys.exit(1)
        
    print(f"Total frame paths found: {len(frame_paths)}")
    frames = load_frames(frame_paths)
    
    if not frames:
        print("Error: Failed to load any frame images.")
        sys.exit(1)
        
    # 2. Extract patch-level features
    print("\nStep 1: Extracting dense patch-level features...")
    if args.encoder == "dinov2":
        encoder = DINOv2Encoder(device=args.device)
    elif args.encoder == "sam":
        if not args.sam_checkpoint:
            print("Error: SAM checkpoint path must be provided via --sam_checkpoint to use SAM encoder.")
            sys.exit(1)
            encoder = SAMEncoder(checkpoint_path=args.sam_checkpoint, device=args.device)
    elif args.encoder == "clip":
        encoder = CLIPEncoder(device=args.device)
    else:
        print(f"Error: Unsupported encoder '{args.encoder}'")
        sys.exit(1)
        
    features = encoder.extract_features(frames, batch_size=args.batch_size)
    print(f"Total patch embeddings extracted: {features.shape[0]} vectors of dimension {features.shape[1]}")
    
    # 3. Compress embeddings with PCA
    print("\nStep 2: Fitting PCA for dimensionality reduction...")
    pca = EmbeddingPCA(n_components=args.pca_dim)
    features_pca = pca.fit_transform(features)
    
    # Save PCA model
    pca_path = os.path.join(args.output_dir, f"{args.encoder}_pca_model.joblib")
    pca.save(pca_path)
    
    # 4. K-Means clustering
    print("\nStep 3: Fitting K-Means clustering...")
    kmeans = FeatureClustering(n_clusters=args.k)
    kmeans.fit(features_pca, subset_ratio=args.subset_ratio)
    
    # Save K-Means model
    kmeans_path = os.path.join(args.output_dir, f"{args.encoder}_kmeans_model.joblib")
    kmeans.save(kmeans_path)
    
    # 5. Generate and save smart cluster visualization
    print("\nStep 4: Creating cluster visualization report...")
    color_map = generate_smart_color_palette(kmeans)
    
    # Select a random frame to visualize
    idx = random.randint(0, len(frames) - 1)
    sample_frame = frames[idx]
    
    # Run prediction on sample frame
    sample_features = encoder.extract_features([sample_frame], batch_size=1, show_progress=False)
    sample_features_pca = pca.transform(sample_features)
    sample_labels = kmeans.predict(sample_features_pca)
    
    # Generate BGR blended visualization map
    color_map_bgr = color_map[..., ::-1]  # Flip to BGR for OpenCV-style image creation
    vis_img = create_cluster_map(
        sample_frame,
        sample_labels,
        color_map_bgr,
        encoder.patch_grid
    )
    
    # Convert BGR back to RGB for matplotlib saving
    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(sample_frame)
    axes[0].set_title("Original Frame")
    axes[0].axis("off")
    
    axes[1].imshow(vis_img_rgb)
    axes[1].set_title("Cluster Map Overlay")
    axes[1].axis("off")
    
    os.makedirs(args.output_dir, exist_ok=True)
    vis_save_path = os.path.join(args.output_dir, f"{args.encoder}_cluster_visualization.jpg")
    plt.tight_layout()
    plt.savefig(vis_save_path, bbox_inches="tight")
    plt.close()
    
    print(f"Sample visualization saved to: {vis_save_path}")
    print("\n=== PIPELINE TRAINING COMPLETE ===")
    print(f"PCA Model: {pca_path}")
    print(f"K-Means Model: {kmeans_path}")

def evaluate_pipeline(args):
    """Runs the quantitative evaluation pipeline (calculates CorLoc)."""
    print("\n=== STARTING EVALUATION PIPELINE ===")
    
    pca_path = os.path.join(args.output_dir, f"{args.encoder}_pca_model.joblib")
    kmeans_path = os.path.join(args.output_dir, f"{args.encoder}_kmeans_model.joblib")
    
    if not os.path.exists(pca_path) or not os.path.exists(kmeans_path):
        print(f"Error: Model files not found at '{pca_path}' or '{kmeans_path}'. Please run training first.")
        sys.exit(1)
        
    print("Loading models...")
    pca = EmbeddingPCA().load(pca_path)
    kmeans = FeatureClustering().load(kmeans_path)
    
    print(f"Initializing encoder '{args.encoder}'...")
    if args.encoder == "dinov2":
        encoder = DINOv2Encoder(device=args.device)
    elif args.encoder == "sam":
        if not args.sam_checkpoint:
            print("Error: SAM checkpoint path must be provided via --sam_checkpoint.")
            sys.exit(1)
        encoder = SAMEncoder(checkpoint_path=args.sam_checkpoint, device=args.device)
    elif args.encoder == "clip":
        encoder = CLIPEncoder(device=args.device)
    else:
        print(f"Error: Unsupported encoder '{args.encoder}'")
        sys.exit(1)
        
    # Run evaluation
    calculate_corloc(
        dataset_dir=args.data_dir,
        class_name=args.category,
        encoder=encoder,
        pca=pca,
        kmeans=kmeans,
        object_cluster=args.object_cluster
    )
    print("\n=== EVALUATION COMPLETE ===")

def main():
    parser = argparse.ArgumentParser(description="Unsupervised Video Object Co-Localization Pipeline")
    parser.add_argument(
        "--action",
        type=str,
        default="train",
        choices=["download", "train", "evaluate", "demo"],
        help="Pipeline action to run (default: 'train')"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="car",
        help="YouTube-Objects class category to process (default: 'car')"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="dinov2",
        choices=["dinov2", "sam", "clip"],
        help="Encoder backbone to ablate (default: 'dinov2')"
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=None,
        help="Path to SAM checkpoint file (required if using --encoder sam)"
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=64,
        help="Target compressed dimensions for PCA (default: 64)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=16,
        help="Number of K-Means clusters (default: 16)"
    )
    parser.add_argument(
        "--subset_ratio",
        type=float,
        default=0.1,
        help="Ratio of features to fit K-Means on (default: 0.1)"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=5,
        help="Step size to sample frames from videos (default: 5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Inference batch size (default: 32)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Root directory where the dataset is stored (default: './data')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dinov2_coloc_output",
        help="Output directory for saved models and visual results (default: 'dinov2_coloc_output')"
    )
    parser.add_argument(
        "--object_cluster",
        type=int,
        default=0,
        help="K-Means Cluster ID corresponding to the target object category for CorLoc evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run torch models on (e.g. 'cuda', 'cpu')"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to launch the Gradio demo application (default: 7860)"
    )
    
    args = parser.parse_args()
    
    if args.action == "download":
        download_and_extract(args.category, args.data_dir)
    elif args.action == "train":
        train_pipeline(args)
    elif args.action == "evaluate":
        evaluate_pipeline(args)
    elif args.action == "demo":
        pca_path = os.path.join(args.output_dir, f"{args.encoder}_pca_model.joblib")
        kmeans_path = os.path.join(args.output_dir, f"{args.encoder}_kmeans_model.joblib")
        launch_demo(pca_path, kmeans_path, port=args.port)

if __name__ == "__main__":
    main()
