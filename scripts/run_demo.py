import os
import cv2
import numpy as np
import warnings
import argparse
import gradio as gr
from coloc.models import DINOv2Encoder
from coloc.pca import EmbeddingPCA
from coloc.clustering import FeatureClustering
from coloc.visualization import generate_smart_color_palette, save_video_with_overlays

# Suppress unpickling warnings for joblib models
warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator")

def launch_demo(pca_path, kmeans_path, port=7860, host="0.0.0.0"):
    """
    Sets up and launches the Gradio web application.
    """
    print("\n--- Initializing Gradio Demo ---")
    print(f"PCA Model Path: {pca_path}")
    print(f"K-Means Model Path: {kmeans_path}")
    
    # 1. Verification of models
    if not os.path.exists(pca_path) or not os.path.exists(kmeans_path):
        print("[Warning] Trained models not found. Please train models first using main.py.")
        print("Demo can still launch, but users will need to specify valid model paths in the UI.")
        
    # Lazy model loading to speed up startup or handle missing files
    models_cache = {}
    
    def get_models(pca_file, kmeans_file):
        cache_key = (pca_file, kmeans_file)
        if cache_key in models_cache:
            return models_cache[cache_key]
            
        if not os.path.exists(pca_file):
            raise gr.Error(f"PCA model file not found at: {pca_file}")
        if not os.path.exists(kmeans_file):
            raise gr.Error(f"K-Means model file not found at: {kmeans_file}")
            
        print("Loading models into memory...")
        encoder = DINOv2Encoder()
        pca = EmbeddingPCA().load(pca_file)
        kmeans = FeatureClustering().load(kmeans_file)
        color_map = generate_smart_color_palette(kmeans)
        
        models_cache[cache_key] = (encoder, pca, kmeans, color_map)
        return models_cache[cache_key]

    def process_videos(video1_path, video2_path, pca_file, kmeans_file, cluster_id, progress=gr.Progress()):
        if not video1_path or not video2_path:
            raise gr.Error("Please upload both video files.")
            
        try:
            encoder, pca, kmeans, color_map = get_models(pca_file, kmeans_file)
        except Exception as e:
            raise gr.Error(f"Error loading models: {str(e)}")
            
        # Parse target cluster ID
        target_cluster = None
        if cluster_id != "All Clusters (Multi-Color)":
            try:
                target_cluster = int(cluster_id)
            except ValueError:
                pass
                
        # Define output names
        dir1, name1 = os.path.split(video1_path)
        dir2, name2 = os.path.split(video2_path)
        
        prefix = f"cluster_{target_cluster}_" if target_cluster is not None else "all_clusters_"
        out1_path = os.path.join(dir1, prefix + name1)
        out2_path = os.path.join(dir2, prefix + name2)
        
        # Process Video 1
        progress(0, desc="Extracting features and generating overlays for Video 1...")
        save_video_with_overlays(
            video1_path, out1_path, encoder, pca, kmeans, color_map,
            object_cluster=target_cluster,
            progress_callback=lambda p: progress(p * 0.5, desc="Processing Video 1...")
        )
        
        # Process Video 2
        progress(0.5, desc="Extracting features and generating overlays for Video 2...")
        save_video_with_overlays(
            video2_path, out2_path, encoder, pca, kmeans, color_map,
            object_cluster=target_cluster,
            progress_callback=lambda p: progress(0.5 + p * 0.5, desc="Processing Video 2...")
        )
        
        progress(1.0, desc="Videos processed successfully!")
        return out1_path, out2_path

    # Define Choices for Cluster Selector (assuming K=16 default)
    cluster_choices = ["All Clusters (Multi-Color)"] + [str(i) for i in range(16)]

    # UI Theme and layout setup
    with gr.Blocks(theme=gr.themes.Soft(), title="Unsupervised Video Object Co-Localization") as demo:
        gr.Markdown(
            """
            # Unsupervised Video Object Co-Localization
            
            This application demonstrates fully unsupervised zero-shot object discovery and co-localization across video sequences. 
            It leverages patch-level embeddings from a frozen **DINOv2** Vision Transformer, compresses them via **PCA**, 
            and groups them using **K-Means Clustering**.
            
            ### Instructions:
            1. Upload two videos containing the common object class (e.g. two videos of cars).
            2. Choose whether to visualize **All Clusters** (semantic segment map) or isolate a **specific Cluster ID** (e.g., target object).
            3. Verify/adjust the Model Paths below.
            4. Click **"Run Co-Localization"** to start.
            """
        )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Videos")
                video_in1 = gr.Video(label="Input Video 1")
                video_in2 = gr.Video(label="Input Video 2")
                
                with gr.Row():
                    pca_model_input = gr.Textbox(label="PCA Model Path", value=pca_path)
                    kmeans_model_input = gr.Textbox(label="K-Means Model Path", value=kmeans_path)
                    
                cluster_select = gr.Dropdown(
                    label="Visualization Target", 
                    choices=cluster_choices, 
                    value="All Clusters (Multi-Color)"
                )
                
                btn = gr.Button("Run Co-Localization", variant="primary")
                
            with gr.Column():
                gr.Markdown("### Co-Localized Outputs")
                video_out1 = gr.Video(label="Output Video 1 (Overlay)")
                video_out2 = gr.Video(label="Output Video 2 (Overlay)")
                
        btn.click(
            fn=process_videos,
            inputs=[video_in1, video_in2, pca_model_input, kmeans_model_input, cluster_select],
            outputs=[video_out1, video_out2]
        )
        
    demo.launch(server_name=host, server_port=port, debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised Co-Localization Gradio Demo")
    parser.add_argument("--pca_path", type=str, default="dinov2_coloc_output/pca_model.joblib", help="Path to trained PCA model")
    parser.add_argument("--kmeans_path", type=str, default="dinov2_coloc_output/kmeans_model.joblib", help="Path to trained K-Means model")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port number")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Gradio host ip")
    
    args = parser.parse_args()
    launch_demo(args.pca_path, args.kmeans_path, port=args.port, host=args.host)
