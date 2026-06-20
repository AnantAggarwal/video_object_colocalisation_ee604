import os
import cv2
import numpy as np
from coloc.data import get_frame_paths, get_annotation_path, parse_annotation

def compute_iou(boxA, boxB):
    """
    Computes Intersection-over-Union (IoU) of two bounding boxes.
    Boxes are in format [xmin, ymin, xmax, ymax].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0.0
        
    return interArea / unionArea

def extract_bbox_from_mask(mask, target_val=1, filter_noise=True):
    """
    Extracts a bounding box [xmin, ymin, xmax, ymax] from a binary mask.
    If filter_noise is True, uses connected component analysis to isolate
    the largest coherent region and ignore scattered noise.
    """
    binary_mask = (mask == target_val).astype(np.uint8)
    if not np.any(binary_mask):
        return None

    if filter_noise:
        # Find all connected components
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
        if num_labels > 1:
            # Skip label 0 (which is the background component)
            sizes = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(sizes) + 1
            x, y, w, h, _ = stats[largest_label]
            return [int(x), int(y), int(x + w), int(y + h)]

    # Fallback to standard bounding box enclosing all active pixels
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [int(xmin), int(ymin), int(xmax), int(ymax)]

def calculate_corloc(dataset_dir, class_name, encoder, pca, kmeans, object_cluster, target_size=224):
    """
    Computes the Correct Localization (CorLoc) score over the dataset.
    CorLoc is defined as the fraction of images/frames where the IoU
    between the predicted bounding box and the ground truth box is >= 0.5.
    """
    print(f"\n--- Starting CorLoc Evaluation for Class: {class_name} ---")
    frame_paths = get_frame_paths(dataset_dir, class_name)
    
    annotated_frames = []
    for path in frame_paths:
        xml_path = get_annotation_path(path)
        if xml_path:
            annotated_frames.append((path, xml_path))
            
    total_annotated = len(annotated_frames)
    if total_annotated == 0:
        print(f"No XML annotations found for {class_name} in {dataset_dir}.")
        return 0.0
        
    print(f"Found {total_annotated} annotated frames to evaluate.")
    
    correct_count = 0
    evaluated_count = 0
    
    for frame_path, xml_path in annotated_frames:
        # Load frame
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        h, w, _ = frame.shape
        
        # Load ground truth boxes
        gt_boxes = parse_annotation(xml_path, target_class=class_name)
        if not gt_boxes:
            continue
            
        evaluated_count += 1
        
        # Extract features
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features = encoder.extract_features([frame_rgb], batch_size=1, show_progress=False)
        
        # Compress and predict
        features_pca = pca.transform(features)
        patch_labels = kmeans.predict(features_pca)
        
        # Reshape to grid and upscale to full resolution
        grid_h, grid_w = encoder.patch_grid
        mask_grid = patch_labels.reshape((grid_h, grid_w))
        mask_full = cv2.resize(mask_grid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Extract predicted bbox for the target cluster
        pred_box = extract_bbox_from_mask(mask_full, target_val=object_cluster, filter_noise=True)
        
        if pred_box is not None:
            # Find the best IoU against all ground truth boxes in the frame
            max_iou = max([compute_iou(pred_box, gt_box) for gt_box in gt_boxes])
            if max_iou >= 0.5:
                correct_count += 1
        else:
            max_iou = 0.0
            
    if evaluated_count == 0:
        print("No frames were successfully evaluated.")
        return 0.0
        
    corloc_score = correct_count / evaluated_count
    print(f"Evaluation Complete.")
    print(f"Total Evaluated: {evaluated_count}")
    print(f"Correct Localizations: {correct_count}")
    print(f"CorLoc Score: {corloc_score * 100:.2f}%")
    return corloc_score
