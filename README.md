# Unsupervised Video Object Co-Localization

This repository contains a modular, high-performance, and fully unsupervised pipeline for Video Object Co-localization (VOCL) and segmentation. The project utilizes state-of-the-art self-supervised Vision Transformers to discover, track, and segment recurring foreground objects across unlabeled video collections without any fine-tuning or human annotations.
 
---

## Key Achievements & Methodology

*   **Objective:** Built a zero-shot, unsupervised video object co-localization/segmentation pipeline on the YouTube-Objects dataset.
*   **Feature Extraction:** Ablated several visual backbones, ultimately deploying a frozen **DINOv2-small ViT** to extract dense patch-level semantic embeddings from video frames.
*   **Dimensionality Reduction:** Compressed 384-dimensional patch embeddings down to **64 dimensions** via Principal Component Analysis (PCA) to filter background noise, resolve feature redundancy, and speed up downstream steps.
*   **Unsupervised Clustering:** Fitted a **K-Means clustering algorithm ($K=16$)** on an optimized 10% randomly sampled global feature matrix to identify and isolate visual concepts (e.g. object parts vs. sky, road, buildings).
*   **Localization & Tracking:** Achieved fully unsupervised discovery and tracking of objects across unseen videos. Applied connected component analysis on the target cluster mask to generate precise bounding boxes and segmentation overlays.

---

## Encoder Ablation Summary

We investigated three different visual backbones for dense feature clustering:

| Architecture / Metric | SAM (Approach 1) | PMC-CLIP (Approach 2) | DINOv2-Small (Final) |
| :--- | :--- | :--- | :--- |
| **Pipeline Overview** | ViT-Base Image Encoder, 64x64 feature map, K-Means | ViT-B Text/Image Encoder, Patch embeddings, K-Means | Self-supervised ViT, 16x16 patch grid, PCA + K-Means |
| **Advantages** | Extremely fine-grained boundaries, high segmentation detail | Strong semantic grounding due to multimodal pre-training | Lightweight, fast inference, robust cross-video temporal consistency |
| **Disadvantages** | Heavy memory usage, slow inference, lack of cross-frame coherence | Domain mismatch (biomedical document captions), high inference overhead | Minor spatial resolution loss compared to SAM |
| **Outcomes** | Discontinued (high compute, low temporal stability) | Discontinued (weak generalization to street-scenes, high latency) | **Selected Backbone** (Optimal trade-off of efficiency, interpretability, and performance) |

---

## Repository Structure

```text
video_object_colocalisation_ee604/
├── data/                         # Local directory for YouTube-Objects dataset
├── notebooks/                    # Archived research notebook workflows
│   ├── data_gathering.ipynb
│   ├── dino-k-mean.ipynb
│   └── sam-embedding.ipynb
├── coloc/                        # Core codebase package
│   ├── data.py                   # Data utilities, path mapping, and XML parsing
│   ├── models.py                 # Encoder backbones (DINOv2, SAM, CLIP)
│   ├── pca.py                    # PCA compression module
│   ├── clustering.py             # K-Means clustering module
│   ├── evaluation.py             # CorLoc metric and IoU calculations
│   └── visualization.py          # Cluster maps, binary overlays, and video writing
├── scripts/                      # Operation scripts
│   ├── download_data.py          # Automates downloading category subsets
│   └── run_demo.py               # Launches Gradio interactive demo app
├── main.py                       # Main pipeline entrypoint (CLI)
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

---

## Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/<your_username>/video_object_colocalisation.git
    cd video_object_colocalisation
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

---

## Operational Guide

All pipeline stages are driven through the unified command-line entrypoint `main.py`.

### 1. Download Dataset Subset
By default, this downloads and extracts the `car` subset of the YouTube-Objects dataset:
```bash
python main.py --action download --category car --data_dir ./data
```

### 2. Train the Pipeline
Extract features using DINOv2-small, perform PCA dimension reduction, and train K-Means:
```bash
python main.py --action train --category car --encoder dinov2 --pca_dim 64 --k 16 --subset_ratio 0.1 --data_dir ./data
```
*This saves the trained model weights in the `dinov2_coloc_output/` folder and generates a sample `dinov2_cluster_visualization.jpg` mapping all 16 clusters.*

### 3. Evaluate CorLoc
To evaluate the Correct Localization (CorLoc) score against standard ground truth XML annotations, identify the target object cluster ID (from the generated visualization) and run:
```bash
python main.py --action evaluate --category car --encoder dinov2 --object_cluster 0 --data_dir ./data
```
*This extracts predicted bounding boxes from the target mask, computes Intersection-over-Union (IoU) with PASCAL VOC ground truth XML coordinates, and reports the final CorLoc metric.*

### 4. Launch the Web Application
Run the interactive Gradio demo to upload and process unseen videos:
```bash
python main.py --action demo --encoder dinov2
```
*Navigate to `http://127.0.0.1:7860` in your browser. Upload two videos containing the common object class to visualize and compare the zero-shot discovery overlays.*
