# Unsupervised Video Object Co-Localization

**Authors:**

* **Anant Aggarwal** (240118) – anantg24@iitk.ac.in
* **Saket Garg** (240907) – saketgarg24@iitk.ac.in
* **Shourya Mathur** (240991) – shouryam24@iitk.ac.in

---

### Abstract

Video Object Co-localization (VOCL) addresses the complex challenge of identifying common object instances across video frames or related videos without supervision. This task is complicated by significant variations in object scale, dynamic lighting conditions, and complex motion. We propose a simple yet powerful self-supervised pipeline that leverages foundational visual models to bypass these challenges. Our method utilizes DINOv2 to extract rich, high-dimensional semantic embeddings from video frames. To reduce noise and isolate salient features from the background, we apply Principal Component Analysis (PCA) for dimensionality reduction. Finally, K-means clustering is employed on the reduced feature space to segment the common foreground object.

Qualitative results demonstrate the pipeline's effectiveness, showing a clear separation of car regions across frames in test sequences, despite viewpoint changes. The novelty of this work lies in its simplicity, providing robust co-localization without requiring any model fine-tuning or bounding box annotations. This outcome highlights the significant and readily transferable semantic understanding embedded within DINOv2 features, proving their utility for dense co-localization tasks. This pipeline offers a simple yet effective baseline for object co-localization from self-supervised features.

---

## 1. Introduction

Video object co-localization is the task of identifying and localizing a common object that appears across multiple unlabeled videos or frames. This problem is of growing importance as vast amounts of video data are captured daily, while detailed annotations remain expensive and time-consuming to produce. It is used in autonomous driving for understanding complex road scenes without labeled data, in medical imaging to enable organ and tumor segmentation with minimal expert annotations, in robotics to let machines recognize and manipulate new objects by learning general visual representations from unlabeled images, and even in security surveillance for capturing recurring suspicious objects. The key insight is that even in the absence of labels, common objects share visual regularities that can be discovered through self-supervised feature learning.

In this work, we present a simple yet effective unsupervised pipeline for video object co-localization using DINOv2, a self-supervised Vision Transformer (ViT) trained to learn rich patch-level representations. Our approach builds upon three main components:

* **Patch-level feature extraction** using a frozen DINOv2 backbone, producing high-dimensional embeddings that capture fine-grained semantics.
* **Dimensionality reduction** via Principal Component Analysis (PCA) to denoise and compress the embeddings.
* **Unsupervised clustering** using K-Means over all patch embeddings to group visually similar regions across frames.

The novelty of this work lies in combining state-of-the-art self-supervised features with a transparent, modular framework that requires no labels, no fine-tuning, and minimal compute. Unlike end-to-end trained models, our approach prioritises interpretability; a dedicated visualisation module allows practitioners to easily identify and extract the common object cluster.

---

## 2. Related Work

The task of Unsupervised Object Discovery (UOD)—localizing objects without human-provided labels—has seen a significant paradigm shift with the advent of self-supervised Vision Transformers (ViTs) [4, 9]. Prior to this, many approaches were weakly-supervised, relying on Class Activation Maps (CAMs) derived from models trained for classification [2]. These CAM-based methods, such as TCAM, suffer from two fundamental limitations. First, as they are products of a classification-driven objective, they are optimized to find only the most discriminative part of an object (e.g., a bird's head) rather than its entire extent [7]. Second, to handle video, they often depend on external and noisy motion cues, like optical flow, which fail in unconstrained videos with camera motion [3, 8].

The introduction of DINO (self-distillation with no labels) [4] demonstrated that self-supervised ViTs possess "emergent properties" for localization; their self-attention maps and features inherently contain explicit, high-resolution information about object boundaries and semantic segments [4]. This breakthrough sparked a new family of UOD methods.

However, many subsequent works have focused on complex, computationally heavy techniques to interpret these rich features. Methods like LOST [10] and TokenCut [11] treat the patch tokens as nodes in a graph and formulate object discovery as a spectral clustering or Normalized Cut problem [11]. While effective, these graph-based optimizations are complex and computationally demanding.

In contrast, our work aligns with a more direct and efficient approach. Research has shown that the powerful, object-centric features from DINO [4] and DINOv2 [9] are so well-structured that they do not require such complex post-processing [1]. It has been demonstrated that simple, lightweight, and zero-shot methods are sufficient to extract high-quality object segments. Specifically, applying Principal Component Analysis (PCA) to DINO patch features has been shown to be a highly effective method for separating foreground and background [5, 6]. Following this dimensionality reduction, standard clustering algorithms, such as K-Means, can be directly applied to the features to group them into semantically coherent object parts [12, 13]. Our method builds on this insight, leveraging a simple pipeline of DINO, PCA, and K-Means to achieve robust object localization and segmentation, demonstrating that the power lies within the features themselves, not in complex optimization.

> **Figure 1. The Overall Architecture:**
> `Video Frames` $\rightarrow$ `DINOv2 (Transformer Encoder)` $\rightarrow$ `Principal Component Analysis` $\rightarrow$ `K-means Clustering` $\rightarrow$ `Common Object`

---

## 3. Our Method

The challenge of video object co-localization is to identify and bound the common object class across a collection of videos without access to any bounding box annotations. Our proposed method addresses this as a zero-shot, unsupervised discovery problem. The core philosophy is to leverage the rich, semantic-level features learned by large-scale foundation models and then, through a multi-stage pipeline, distill these features into concrete spatial localizations.

Our pipeline operates on the principle that visually similar semantic concepts (e.g., "car," "road," "sky") will form distinct, separable clusters in a high-dimensional feature space. By identifying the cluster corresponding to the common object, we can generate localization maps for the entire video collection. The method is comprised of four primary stages, as illustrated in our framework: (1) Patch-level feature extraction, (2) Dimensionality reduction, (3) Unsupervised feature clustering, and (4) Localization map generation.

### 3.1. Feature Extraction with DINOv2

The foundation of our approach is a powerful, pre-trained vision model. We utilize DINOv2 (Self-Distillation with NO labels, v2), specifically the ViT-Small backbone (`dinov2-small`). Unlike models trained with supervised classification, DINOv2 is trained via self-supervision. It employs a student-teacher network architecture, where a student network is trained to match the output distribution of a teacher network (which is an exponential moving average of the student's own weights, a "momentum encoder").

The model is trained by being fed multiple, differently-augmented "views" of the same image. The student is trained on small, local "patches" of the image and must match the teacher's output for larger, global "views." This self-distillation objective, optimized with a cross-entropy loss, forces the model to learn high-level, semantic representations without any human-provided labels.

A key emergent property of this training scheme is that the model's patch-level embeddings become semantically rich. While traditional ViTs are optimized for the global `[CLS]` token for classification, DINOv2's patch tokens are trained to encode visual semantics. Patches corresponding to the same object class (e.g., a "car door" and a "car hood") will have highly similar feature vectors, distinct from "background" patches like "road" or "sky."

For our implementation, we process each frame from the car subset of the YouTube-Objects dataset. Each frame is passed through the DINOv2 model, and we extract the final hidden-state output for all patch tokens, effectively creating a 2D grid of high-dimensional feature vectors for every frame.

### 3.2. Dimensionality Reduction via PCA

The output from the DINOv2-small model is a 384-dimensional feature vector for each patch. This high dimensionality is computationally expensive for downstream clustering and likely contains significant redundant information, where multiple feature channels are highly correlated.

To address this, we employ Principal Component Analysis (PCA). PCA is a linear transformation technique used to find a new, lower-dimensional basis for a set of data. It operates by identifying the directions, or "principal components," along which the data has the maximum variance.

Let our entire set of extracted patch features (from all frames in the dataset) be represented by a matrix $X$, where each row is a 384-dimensional feature vector. PCA first computes the covariance matrix of the (mean-centered) data:

$$C = \frac{1}{n-1} X^T X$$

The principal components are the eigenvectors of this covariance matrix $C$. The corresponding eigenvalues represent the amount of variance captured by each eigenvector. By selecting the $k$ eigenvectors with the largest eigenvalues, we create a projection matrix $W$. We then project our original data $X$ onto this new, $k$-dimensional subspace:

$$X_{\text{reduced}} = XW$$

In our work, we reduce the feature space from 384 dimensions to $k=64$. This $k$-dimensional space retains the vast majority of the data's variance while discarding redundant information and noise, leading to a more robust and efficient clustering process.

### 3.3. Unsupervised Feature Clustering

Once we have a collection of reduced-dimension patch features, our goal is to group them into semantically meaningful "concepts." We hypothesize that all patches belonging to the "car" object class will form a distinct cluster.

We employ the K-Means clustering algorithm for this task. K-Means is an iterative algorithm that partitions a dataset into $K$ pre-defined, non-overlapping clusters. It aims to minimize the within-cluster sum of squares (i.e., the variance within each cluster).

We first aggregate all PCA-reduced patch features from a large subset of video frames into a single set. We then fit a K-Means model to this data with a fixed number of clusters, $K=16$. Setting $K > 2$ is a deliberate choice; it allows the model to find distinct clusters for the primary object (car) as well as for various common background elements (e.g., "road," "sky," "buildings," "trees").

The result of this stage is a fitted K-Means model that acts as a "concept quantizer." For any given patch feature, the model can now assign it a cluster ID representing its learned semantic group.

### 3.4. Localization and Visualization Map Generation

The final stage synthesizes the previous steps to generate spatial localization maps. This process is bifurcated, producing two distinct outputs: (1) a qualitative, multi-cluster visualization map for human analysis and (2) a quantitative, single-cluster binary mask used for performance evaluation.

#### 3.4.1 Qualitative Visualization Map

To intuitively understand the complete semantic segmentation performed by the K-Means algorithm, we generate a comprehensive "cluster map." This map visualizes all $K$ clusters simultaneously.

A key aspect of this visualization is a color scheme designed to reflect the similarity of the clusters themselves. This color map is generated by treating the $K$ cluster centroids—which are $k=64$ dimensional points in the PCA-reduced feature space—as a new, small dataset. We apply a secondary PCA transformation to project these $K$ centroids from 64D down to 3D. These 3D coordinates are then normalized to a $[0, 1]$ range using a Min-Max scaler, effectively mapping each cluster center to a unique (R, G, B) color value.

The result is a perceptually-aware color map. Clusters that are "closer" in the 64D feature space (i.e., more semantically similar) are assigned more "similar" RGB colors.

For any given frame, we first generate its low-resolution (e.g., $16\times16$) cluster-ID map by applying the fitted PCA and K-Means models to its patch features. We then use our generated color map to "paint" each pixel of this low-resolution map according to its cluster ID. This color map is upscaled (via nearest-neighbor interpolation) to the original frame's resolution and blended with the original image. This provides a single, comprehensive visualization of all learned semantic concepts (e.g., "car body," "wheels," "road," "sky") in the frame.

> **Figure 2. Qualitative Results:**
> Side-by-side comparison of `Original` and `Cluster Map` across various frames showing that the red cluster is automatically associated with a car.

#### 3.4.2 Quantitative Localization Mask

For quantitative evaluation, such as calculating the CorLoc score, a single binary mask for the target object is required. This first necessitates identifying the specific cluster ID that corresponds to the common object class. Because our method is fully unsupervised, we do not know a priori which cluster ID corresponds to the "car."

We identify this object cluster, $c_{\text{obj}}$, by visually inspecting the qualitative cluster maps generated across the dataset. The cluster that consistently highlights the common object with the largest and most central activation area—in this case, the 'car'—is selected.

Once $c_{\text{obj}}$ is identified, we generate a binary mask for each frame. We take the $16\times16$ cluster-ID map and set all pixels with the label $c_{\text{obj}}$ to 1 and all other pixels to 0. This binary mask is then upscaled to the original frame's resolution, creating a final localization mask. This mask is then used to derive a bounding box, which is compared against the ground truth to compute our evaluation metrics.

---

## 4. Experiments

### 4.1. First Approach: SAM (Segment Anything Model)

*Link to the research paper: "Segment Anything," Meta AI Research, 2023.*

#### 4.1.1 Summary of the Approach

Our initial attempt focused on leveraging the powerful segmentation capabilities of the Segment Anything Model (SAM). Instead of using SAM in its prompt-based segmentation mode, we experimented with using only its image encoder to extract dense patch-level embeddings from video frames. The hypothesis was that SAM's large-scale pretraining on the SA-1B dataset (over 1.1B masks) would provide feature representations that naturally separate objects from the background, enabling object co-localization in a fully unsupervised setting.

#### 4.1.2 Motivation

Since SAM is trained to segment any object within a single image, we expected its internal feature space to already encode object boundaries and semantic cues. The intuition was that clustering these embeddings across frames would allow us to discover the recurrent object (the car) without supervision or fine-tuning.

#### 4.1.3 Experimental Progress

Rather than implementing the full pipeline immediately, we adopted a progressive, trial-and-error workflow:

* Extracted frame-wise dense embeddings using SAM's ViT encoder.
* Reduced embedding dimensionality using PCA (64, 128, and 256 dimensions tested).
* Applied K-Means clustering ($K=3$ to $8$) to study the emergence of object clusters.
* Overlaid cluster assignments on frames to visually inspect object-level consistency.

Across multiple runs, one of the clusters consistently highlighted the car body, while the remaining clusters corresponded to background elements such as sky, road, or shadows. This confirmed that SAM's feature space is discriminative even without supervision.

#### 4.1.4 Key Points of the Approach

* **Dense Feature Extraction:** SAM's ViT encoder produces high-dimensional pixel embeddings containing semantic and boundary information.
* **Fully Unsupervised:** No labels, prompts, or masks were used; clustering operated purely on encoder outputs.
* **Hyperparameter Sweeps:** PCA dimension and cluster count were varied to study stability, compactness, and object coverage.

#### 4.1.5 Observations

* One cluster consistently captured the car foreground region.
* However, the feature alignment was frame-wise only, resulting in unstable cluster assignments across time.

#### 4.1.6 Disadvantages

* **Heavy Computation:** SAM's ViT encoder is GPU and memory intensive, making large-scale video processing slow.
* **No Temporal Coherence:** Trained for static image segmentation, SAM lacks constraints for cross-frame consistency.
* **Inefficient Pipeline:** Even after PCA compression, storing and clustering dense embeddings was computationally expensive.

#### 4.1.7 Process Overview

| Step | Description |
| --- | --- |
| 1 | Extract dense embeddings for each frame using SAM's image encoder. |
| 2 | Apply PCA (64-256D) to reduce feature dimensionality. |
| 3 | Perform K-Means clustering on patch embeddings. |
| 4 | Compute proxy metrics (cluster inertia, coverage). |
| 5 | Visualize cluster overlays to identify the co-localized region. |

**Table 1. Process for Co-localized Region Identification**

#### 4.1.8 Reason for Discontinuation

Although SAM provided clean within-frame segmentation cues, the approach was ultimately discontinued due to its high computational cost and weak temporal consistency. These limitations motivated a shift toward lighter encoders with better semantic alignment across images, leading to the transition to a DINOv2-based clustering framework.

### 4.2. Second Approach: PMC-CLIP (Contrastive Language-Image Pre-training)

*Link to the research paper: "PMC-CLIP: Contrastive Language-Image Pre-training using Biomedical Documents," 2023.*

#### 4.2.1 Summary of the Approach

After experimenting with purely vision-based encoders, we explored whether a multimodal model such as PMC-CLIP could offer stronger semantic separation of objects. PMC-CLIP is a CLIP-style model trained on 1.6M biomedical image-text pairs (PMC-OA), where images are aligned with their corresponding scientific captions using contrastive learning.

Our motivation was that such multimodal training might yield embeddings that are richer and more object-aware than standard image-only encoders. We therefore used PMC-CLIP's image encoder to extract dense frame embeddings and performed unsupervised clustering, similar to the SAM pipeline.

#### 4.2.2 Motivation

Unlike generic visual encoders, PMC-CLIP learns from paired images and captions, meaning its embedding space is organized not only by visual similarity but also by high-level semantic associations. We wanted to test whether this semantic prior could help improve object co-localization across videos, even though the model was originally trained on biomedical data.

#### 4.2.3 Experimental Progress

The workflow was largely iterative:

* Extracted frame-level patch embeddings using PMC-CLIP's encoder.
* Applied PCA compression (64, 96, 128 dimensions tested) to reduce memory and speed up clustering.
* Ran K-Means with varying $K$ values to study whether object patches formed distinct clusters.
* Overlaid cluster assignments on frames to check if the recurring object (car) emerged.

Compared to SAM, PMC-CLIP embeddings showed stronger semantic grouping: foreground regions were more tightly clustered, and background noise was lower. However, the clusters still shifted across frames, indicating limited temporal stability.

#### 4.2.4 Key Points of the Approach

* **Multimodal Pretraining:** Trained on 1.6M image-caption pairs, providing stronger semantic grounding than vision-only encoders.
* **CLIP-Style Alignment:** Contrastive learning aligns visual and textual feature spaces, improving semantic separability.
* **Unsupervised Co-localization:** Patch embeddings were clustered ($\text{PCA} + \text{K-Means}$) to find the recurrent object region across videos.
* **Observation:** Clusters were cleaner than SAM in terms of visual semantics, but still not stable enough across frames.

#### 4.2.5 Disadvantages

* **Domain Mismatch:** The model was trained on biomedical figures and captions, so its features did not fully generalize to street-scene videos.
* **High Inference Cost:** The encoder is large and slow when applied patch-wise on every frame of long videos.
* **Limited Temporal Consistency:** Despite richer embeddings, cross-video alignment still required manual inspection and tuning.

#### 4.2.6 Process Overview

| Step | Description |
| --- | --- |
| 1 | Extract patch embeddings from frames using PMC-CLIP's image encoder. |
| 2 | Reduce embedding dimensionality using PCA (64-128D). |
| 3 | Perform K-Means clustering on all patch embeddings. |
| 4 | Compute proxy metrics (cluster compactness, coverage). |
| 5 | Overlay clusters on frames to identify the recurring object region. |

**Table 2. Pipeline of the PMC-CLIP based co-localization experiment.**

#### 4.2.7 Reason for Discontinuation

Although PMC-CLIP provided semantically richer features than SAM, it suffered from two key limitations: (1) high computational overhead during dense inference, and (2) weak generalization outside the biomedical domain. Because of this, we shifted toward a more efficient and domain-agnostic strategy using DINOv2 embeddings with $\text{PCA} + \text{K-Means}$, which offered a better trade-off between semantic quality and computational cost.

---

### Architectural Comparison Summary

| Metric / Aspect | SAM (Approach 1) | PMC-CLIP (Approach 2) | Ours (DINOv2) (Final) |
| --- | --- | --- | --- |
| **Pipeline Overview** | ViT encoder, Dense features, Clustering | Text/Image Encoder, Patch embeddings, Affinity Map | Self-supervised ViT, PCA compression, K-Means grouping |
| **Cons / Drawbacks** | Heavy model, Slow inference | Requires text prompts, Less boundary precision | Minor resolution loss |
| **Pros / Strengths** | High segmentation detail | Zero-shot capability, Strong concept grounding | Lightweight, Interpretable, Consistent |

> **Figure 3.** Comparison of three approaches for video object co-localization. Our final DINOv2-based pipeline achieves the best balance of efficiency, interpretability, and semantic consistency.

---

## 5. Final Approach and Conclusion

We investigated three different strategies for unsupervised video object co-localization: SAM-based dense feature clustering, multimodal PMC-CLIP embeddings, and finally a compact DINOv2-based pipeline. Each iteration clarified both the strengths and the practical limitations of increasingly complex models when applied in a fully label-free setting.

The SAM approach delivered precise pixel-level embeddings, but its computational cost and lack of temporal coherence made it unsuitable for large-scale video processing. PMC-CLIP improved semantic grouping, yet its biomedical-domain pretraining and heavy vision-language architecture resulted in slow inference and weak generalization to natural scenes.

The final $\text{DINOv2} + \text{PCA} + \text{K-Means}$ method provided the most stable and efficient outcome. Patch-level DINOv2 features consistently isolated the recurring object across frames while remaining lightweight enough to process full videos without GPU overhead. The generated overlays were clean, interpretable, and required neither manual labels nor post-processing.

Overall, the experiments show that self-supervised visual backbones such as DINOv2 strike the best balance between semantic quality, scalability, and simplicity—making them well suited for practical, real-world co-localization pipelines without annotations.

---

## References

1. Amir, A., et al. (2021). Deep ViT Features as Dense Visual Descriptors. *arXiv:2112.05814*.
2. Belharbi, S., Ben Ayed, I., McCaffrey, L., & Granger, E. (2023). TCAM: Temporal Class Activation Maps for Object Localization in Weakly-Labeled Unconstrained Videos. In *WACV*.
3. Belharbi, S., Ben Ayed, I., McCaffrey, L., & Granger, E. (2023). TCAM: Temporal Class Activation Maps for Object Localization in Weakly-Labeled Unconstrained Videos. In *WACV*.
4. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging Properties in Self-Supervised Vision Transformers. In *ICCV*.
5. Hamilton, M., Zhang, Z., Hariharan, B., Snavely, N., & Freeman, W. T. (2022). Unsupervised Semantic Segmentation by Distilling Feature Correspondences. In *ICLR*.
6. Liao, G., et al. (2024). Weakly-supervised Contrastive Learning for Unsupervised Object Discovery. *arXiv:2403.07700*.
7. Lu, Z., et al. (2019). See More, Know More: Unsupervised Video Object Segmentation With Co-Attention. In *CVPR*.
8. Murtaza, S., et al. (2024). Transformer-Based Class Activation Mapping for Weakly-Supervised Video Object Localization. *arXiv:2407.06018*.
9. Oquab, M., Darcet, T., Moutakanni, T., Vo, H. V., Szafraniec, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. In *TMLR*.
10. Siméoni, O., Puy, G., Vo, H. V., Roburin, S., Gidaris, S., et al. (2021). Localizing Objects with Self-Supervised Transformers and no Labels. In *BMVC*.
11. Wang, Y., Shen, X., Hu, S. X., Yuan, Y., Crowley, J. L., & Vaufreydaz, D. (2022). Self-supervised Transformers for Unsupervised Object Discovery using Normalized Cut. In *CVPR*.
12. Zhang, Z., et al. (2022). K-Means Clustering on Self-Supervised ViT Features for Object Segmentation.
13. Ziegler, K., et al. (2022). Self-Supervised Learning of Object Parts for Semantic Segmentation. In *CVPR*.

---

## Appendix A. Implementation Snippets

*Note: The code snippets given in this appendix are only for reference and may not run as is.*

### A.1. Frame Enumeration and Loading

```python
import glob
from tqdm import tqdm
import cv2

frame_paths = glob.glob("/kaggle/input/car-yt-obj-dataset/car/data/000-/shots/0/Frame*.jpg")

def get_frames(frame_paths):
    frames = []
    for path in tqdm(frame_paths, total=len(frame_paths)):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    return frames

# Listing 1. Frame discovery and loading.

```

### A.2. DINOv2 Feature Extraction

```python
import os
import torch
from transformers import AutoImageProcessor, AutoModel

OUTPUT_DIR = "dinov_coloc_output"
MODEL_NAME = "facebook/dinov2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def extract_features(frames, model, processor):
    # Extracts DINOv2 patch features for a batch of frames
    if not frames:
        return None
    inputs = processor(images=frames, return_tensors="pt", do_resize=True, size=224).to(DEVICE)
    outputs = model(**inputs)
    patch_features = outputs.last_hidden_state[:, 1:, :]
    num_frames, num_patches, dim = patch_features.shape
    return patch_features.reshape(num_frames * num_patches, dim).cpu()

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")
print("Loading DINOv2 model and processor...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

patch_size = model.config.patch_size
print(processor.size)
grid_size = 224 // patch_size
PATCH_GRID = (grid_size, grid_size)

print("Part 1: Extracting Features...")
all_features_list = []

for class_name in ['car']:
    print(f"Processing class: {class_name}")
    # assuming frame_paths is defined
    frames = get_frames(frame_paths[::5])
    if not frames:
        continue
    # Batch processing for efficiency
    batch_size = 32
    for i in tqdm(range(0, len(frames), batch_size), desc=f"Extracting {class_name}"):
        batch_frames = frames[i:i+batch_size]
        features = extract_features(batch_frames, model, processor)
        if features is not None:
            all_features_list.append(features)

all_features = torch.cat(all_features_list, dim=0).numpy()
# Listing 2. Patch features with DINOv2.

```

### A.3. PCA + K-Means

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import numpy as np

PCA_COMPONENTS = 64
N_CLUSTERS = 16
KMEANS_SUBSET_RATIO = 0.1

PCA_MODEL_PATH = os.path.join(OUTPUT_DIR, 'pca_model.joblib')
KMEANS_MODEL_PATH = os.path.join(OUTPUT_DIR, 'kmeans_model.joblib')

print("Part 2: Running PCA and K-Means")
print(f"Running PCA to reduce to {PCA_COMPONENTS} dimensions...")

pca = PCA(n_components=PCA_COMPONENTS)
features_pca = pca.fit_transform(all_features)

print(f"Training K-Means on a {KMEANS_SUBSET_RATIO * 100}% subset of features...")
n_samples = features_pca.shape[0]
subset_indices = np.random.choice(n_samples, int(n_samples * KMEANS_SUBSET_RATIO), replace=False)
features_subset = features_pca[subset_indices]

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
kmeans.fit(features_subset)

print("Saving PCA and K-Means models...")
joblib.dump(pca, PCA_MODEL_PATH)
joblib.dump(kmeans, KMEANS_MODEL_PATH)

# Listing 3. Dimensionality reduction and clustering.

```

### A.4. Visualization

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import cv2
import numpy as np

VIS_FILE_PATH = os.path.join(OUTPUT_DIR, 'cluster_visualization.png')

def visualize_all_clusters_one_image(frame, patch_labels, color_map, patch_grid_size, target_size):
    frame_resized = cv2.resize(frame, (target_size, target_size))
    frame_float = frame_resized.astype(np.float32) / 255.0
    mask = patch_labels.reshape(patch_grid_size)
    mask_rgb_lowres = np.zeros((*patch_grid_size, 3), dtype=np.float32)
    
    for k in range(len(color_map)):
        mask_rgb_lowres[mask == k] = color_map[k]
        
    mask_rgb_full = cv2.resize(mask_rgb_lowres, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    blended_img = cv2.addWeighted(frame_float, 0.6, mask_rgb_full, 0.4, 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(frame_resized)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(blended_img)
    axes[1].set_title("Cluster Map")
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

print("Part 3: Generating Cluster Visualization")
print("Generating color map from K-Means centroids...")

centroids_pca = kmeans.cluster_centers_
pca_color = PCA(n_components=3)
centroids_3d = pca_color.fit_transform(centroids_pca)
scaler = MinMaxScaler()
color_map = scaler.fit_transform(centroids_3d)

print(f"Generated color map of shape: {color_map.shape}")
print("Finding one sample frame from each class...")

sample_frames = []
if frames:
    sample_frames.append(frames[np.random.randint(len(frames))])

sample_features = extract_features(sample_frames, model, processor).numpy()
sample_features_pca = pca.transform(sample_features)
sample_labels = kmeans.predict(sample_features_pca)

patches_per_frame = PATCH_GRID[0] * PATCH_GRID[1]
print(f"Sample features shape: {sample_features.shape}")

figs = []
for i, frame in enumerate(sample_frames):
    start_idx = i * patches_per_frame
    end_idx = (i + 1) * patches_per_frame
    patch_labels = sample_labels[start_idx:end_idx]
    
    fig = visualize_all_clusters_one_image(
        frame,
        patch_labels,
        color_map,
        PATCH_GRID,
        224
    )
    fig.suptitle(f"class: car", fontsize=16, y=1.02)
    figs.append(fig)

combined_fig = plt.figure(figsize=(10, 5 * len(figs)))
for i, fig in enumerate(figs):
    ax = combined_fig.add_subplot(len(figs), 1, i + 1)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    ax.imshow(img)
    ax.axis('off')
    plt.close(fig)

plt.tight_layout()
plt.savefig(VIS_FILE_PATH)
print(f"Visualization saved to: {VIS_FILE_PATH}")
print("!!! ACTION REQUIRED !!!")
print(f"Open {VIS_FILE_PATH} and identify the cluster ID for your co-located object.")
print("The colors in the 'Cluster Map' will not match the final red overlay plot, but the cluster IDs are the same.")
print("Then, call the 'calculate_corloc_score' function with that ID.")

# Listing 4. Cluster overlay visualization.

```

---

## Appendix B. Links

* The `car.tar.gz` dataset was used from the following: **YouTube-Objects Dataset (ETH Zurich)**.
* Link to GitHub repository: **Video Object Colocalisation Code**.