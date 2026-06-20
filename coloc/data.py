import os
import glob
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

def get_frame_paths(dataset_dir, class_name="car", sample_rate=1):
    """
    Finds all frame paths for a given category inside the dataset directory.
    Uses the standard YouTube-Objects layout:
    {dataset_dir}/{class_name}/data/000*/shots/0*/frame*.jpg
    """
    pattern = os.path.join(dataset_dir, class_name, "data", "000*", "shots", "0*", "frame*.jpg")
    frame_paths = glob.glob(pattern)
    
    # Fallback to a case-insensitive recursive search if standard layout is not found
    if not frame_paths:
        search_pattern = os.path.join(dataset_dir, "**", "*.jpg")
        frame_paths = glob.glob(search_pattern, recursive=True)
        frame_paths = [p for p in frame_paths if class_name.lower() in p.lower()]
        
    frame_paths = sorted(frame_paths)
    if sample_rate > 1:
        frame_paths = frame_paths[::sample_rate]
        
    return frame_paths

def load_frames(frame_paths, show_progress=True):
    """
    Loads images from the given paths, resizes them (optional), and converts to RGB.
    """
    frames = []
    iterator = tqdm(frame_paths, desc="Loading frames") if show_progress else frame_paths
    for path in iterator:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
    return frames

def get_annotation_path(frame_path):
    """
    Maps a frame image path to its corresponding XML annotation file.
    Supports in-place annotations (sharing the same folder) or parallel annotations folders.
    """
    # 1. Check in-place: replacing image extension with .xml
    base, ext = os.path.splitext(frame_path)
    xml_path = base + ".xml"
    if os.path.exists(xml_path):
        return xml_path
        
    # 2. Check parallel structure: replacing "/data/" with "/annotations/"
    # e.g., /car/data/0007/shots/018/frame0001.jpg -> /car/annotations/0007/shots/018/frame0001.xml
    parts = frame_path.split(os.sep)
    try:
        data_idx = len(parts) - 1 - parts[::-1].index("data")
        parts[data_idx] = "annotations"
        parallel_path = os.sep.join(parts)
        parallel_xml_path = os.path.splitext(parallel_path)[0] + ".xml"
        if os.path.exists(parallel_xml_path):
            return parallel_xml_path
    except ValueError:
        pass
        
    return None

def parse_annotation(xml_path, target_class="car"):
    """
    Parses a YouTube-Objects PASCAL VOC XML annotation file.
    Returns a list of bounding boxes: [[xmin, ymin, xmax, ymax], ...]
    """
    if not xml_path or not os.path.exists(xml_path):
        return []
        
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return []
        
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name")
        if name is not None and name.text.lower() == target_class.lower():
            bbox = obj.find("bndbox")
            if bbox is not None:
                try:
                    xmin = int(float(bbox.find("xmin").text))
                    ymin = int(float(bbox.find("ymin").text))
                    xmax = int(float(bbox.find("xmax").text))
                    ymax = int(float(bbox.find("ymax").text))
                    boxes.append([xmin, ymin, xmax, ymax])
                except (ValueError, TypeError, AttributeError):
                    continue
    return boxes
