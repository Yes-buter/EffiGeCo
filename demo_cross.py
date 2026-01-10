from torch.nn import DataParallel
from models.geco_infer import build_model
from utils.arg_parser import get_argparser
import argparse
import torch
from torchvision import transforms as T
import matplotlib.patches as patches
from PIL import Image
from torchvision import ops
from utils.data import resize_and_pad
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None
    filedialog = None

bounding_boxes = []


# Global variables to track drawing state
rect = None
start_x, start_y = None, None

# Event handler for mouse press (start drawing)
def on_press(event):
    global start_x, start_y, rect
    if event.inaxes:
        start_x, start_y = event.xdata, event.ydata  # Store starting point
        # Create a rectangle (but do not draw yet)
        rect = patches.Rectangle((start_x, start_y), 0, 0, linewidth=2, edgecolor='r', facecolor='none')
        event.inaxes.add_patch(rect)
        plt.draw()  # Update plot to show rectangle (even if not yet drawn)

# Event handler for mouse motion (while drawing)
def on_motion(event):
    global start_x, start_y, rect
    if rect is not None and event.inaxes:
        # Update the width and height of the rectangle based on mouse position
        width = event.xdata - start_x
        height = event.ydata - start_y
        rect.set_width(width)
        rect.set_height(height)
        plt.draw()  # Redraw to update the rectangle while dragging

# Event handler for mouse release (end drawing)
def on_release(event):
    global rect
    # Once mouse is released, we finalize the bounding box
    if rect is not None:
        bounding_boxes.append([rect.get_x(), rect.get_y(), rect.get_x() + rect.get_width(), rect.get_y() + rect.get_height()])
        rect = None  # Reset rect after release


def _choose_image_file(title: str):
    """Open a native file picker dialog to select an image file."""
    if tk is None or filedialog is None:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    try:
        path = filedialog.askopenfilename(
            title=title,
            initialdir=os.getcwd(),
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.webp"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()

    return path if path else None


@torch.no_grad()
def demo_cross(args):
    bounding_boxes.clear()

    support_img_path = getattr(args, 'support_image_path', None)
    query_img_path = getattr(args, 'query_image_path', None)

    if not support_img_path or not os.path.exists(support_img_path):
        picked = _choose_image_file("Select SUPPORT image")
        if picked:
            support_img_path = picked

    if not query_img_path or not os.path.exists(query_img_path):
        picked = _choose_image_file("Select QUERY image")
        if picked:
            query_img_path = picked

    if not support_img_path or not query_img_path:
        print("No image selected. Exiting.")
        return
    
    global fig, ax

    gpu = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device = torch.device(gpu)
    else:
        device = torch.device("cpu")

    model = DataParallel(
        build_model(args).to(device),
        device_ids=[gpu] if torch.cuda.is_available() else None,
        output_device=gpu if torch.cuda.is_available() else None
    )
    
    # Load weights
    try:
        checkpoint = torch.load('GeCo.pth', map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: GeCo.pth not found. Please ensure the model weights are in the current directory.")
        return

    model.eval()

    # --- Step 1: Process Support Image ---
    print(f"Loading support image: {support_img_path}")
    support_image_pil = Image.open(support_img_path).convert("RGB")
    support_image_tensor = T.ToTensor()(support_image_pil)

    # Interactive plotting for support image
    fig, ax = plt.subplots(1)
    ax.imshow(support_image_pil)
    plt.axis('off')
    
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    plt.title("Draw a box on the SUPPORT image, then close window")
    plt.show()

    if not bounding_boxes:
        print("No bounding box drawn. Exiting.")
        return

    # Prepare support data
    support_bboxes = torch.tensor(bounding_boxes, dtype=torch.float32)
    
    # Resize and pad support image and boxes
    # Note: resize_and_pad expects (C, H, W) and (N, 4)
    supp_img_resized, supp_bboxes_resized, supp_scale = resize_and_pad(support_image_tensor, support_bboxes, full_stretch=False)
    
    # Normalize
    supp_img_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(supp_img_resized).unsqueeze(0).to(device)
    supp_bboxes_norm = supp_bboxes_resized.unsqueeze(0).to(device)

    # --- Step 2: Process Query Image ---
    print(f"Loading query image: {query_img_path}")
    query_image_pil = Image.open(query_img_path).convert("RGB")
    query_image_tensor = T.ToTensor()(query_image_pil)
    
    # Dummy boxes for query image resizing (we don't have boxes yet)
    dummy_bboxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
    query_img_resized, _, query_scale = resize_and_pad(query_image_tensor, dummy_bboxes, full_stretch=False)
    
    query_img_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(query_img_resized).unsqueeze(0).to(device)

    # --- Step 3: Cross-Image Inference ---
    print("Running cross-image inference...")
    # Access the underlying model if wrapped in DataParallel
    if isinstance(model, DataParallel):
        model_core = model.module
    else:
        model_core = model

    outputs, _, _, _, masks = model_core.forward_cross(supp_img_norm, supp_bboxes_norm, query_img_norm)
    
    # --- Step 4: Post-processing and Visualization ---
    idx = 0
    thr = 4
    
    # Filter boxes
    keep = ops.nms(outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr],
                   outputs[idx]['box_v'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr], 0.5)

    boxes = (outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr])[keep]
    
    # Clamp boxes to [0, 1]
    final_bboxes = torch.clamp(boxes, 0, 1)
    
    # Convert back to original image coordinates
    # The output boxes are normalized [0, 1] relative to the resized/padded image (1024x1024 usually)
    # We need to map them back to the original query image size
    
    # Get dimensions
    orig_w, orig_h = query_image_pil.size
    resized_h, resized_w = query_img_resized.shape[1], query_img_resized.shape[2]
    
    # Scale boxes to resized image dimensions
    final_bboxes[:, 0] *= resized_w
    final_bboxes[:, 1] *= resized_h
    final_bboxes[:, 2] *= resized_w
    final_bboxes[:, 3] *= resized_h
    
    # Undo padding and scaling
    # resize_and_pad logic: 
    # 1. Scale image so longest side is target_size (e.g. 1024)
    # 2. Pad to square
    
    # We can use the scale factor returned by resize_and_pad
    # But we need to know the padding. 
    # Let's look at resize_and_pad implementation or infer it.
    # Usually it pads right and bottom.
    
    # Let's just visualize on the resized image for simplicity, or try to map back.
    # To map back: divide by scale.
    final_bboxes /= query_scale
    
    # Visualize results on Query Image
    plt.figure(figsize=(10, 10))
    plt.imshow(query_image_pil)
    ax = plt.gca()
    
    for box in final_bboxes:
        box = box.cpu().numpy()
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        
    plt.title(f"Detection Results on Query Image (Found {len(final_bboxes)} objects)")
    plt.axis('off')
    plt.show()


def run_batch_inference(model, support_img_pil, support_box, query_imgs_pil, device):
    """
    Batch cross-image inference.
    
    Args:
        model: Loaded GeCo model
        support_img_pil: Support image (PIL Image)
        support_box: Box on support image [x1, y1, x2, y2] OR polygon points
        query_imgs_pil: List of query images [PIL Image, ...]
        device: torch device
        
    Returns:
        results: List of lists of boxes.
                 results[0] is the support box [[x1, y1, x2, y2]]
                 results[1:] are boxes for each query image [[x1, y1, x2, y2, score], ...]
    """
    results = []
    
    # Preprocess support
    support_image_tensor = T.ToTensor()(support_img_pil)
    
    # Handle various box formats (bbox or polygon points)
    # User format might be: [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
    try:
        box_data = np.array(support_box)
        if box_data.size > 0 and box_data.shape[-1] == 2: # Contains points (x, y)
            # Flatten all points and find min/max to get bounding box
            points = box_data.reshape(-1, 2)
            x_min = points[:, 0].min()
            y_min = points[:, 1].min()
            x_max = points[:, 0].max()
            y_max = points[:, 1].max()
            support_bboxes = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
        else:
            # Assume it's already bbox format [x1, y1, x2, y2]
            support_bboxes = torch.tensor(support_box, dtype=torch.float32)
            if support_bboxes.dim() == 1:
                support_bboxes = support_bboxes.unsqueeze(0)
            elif support_bboxes.dim() > 2:
                support_bboxes = support_bboxes.view(-1, 4)
    except Exception as e:
        print(f"Error processing support_box: {e}")
        # Fallback to original logic if numpy conversion fails
        support_bboxes = torch.tensor([support_box], dtype=torch.float32)

    # 1. Record support result (standardized) with dummy score 1.0
    # support_bboxes is [1, 4]
    support_score = torch.tensor([[1.0]], dtype=torch.float32)
    support_result = torch.cat([support_bboxes, support_score], dim=1)
    results.append(support_result.cpu().numpy().tolist())
    
    supp_img_resized, supp_bboxes_resized, supp_scale = resize_and_pad(
        support_image_tensor, support_bboxes, full_stretch=False
    )
    supp_img_norm = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )(supp_img_resized).unsqueeze(0).to(device)
    supp_bboxes_norm = supp_bboxes_resized.unsqueeze(0).to(device)

    # Handle DataParallel
    if isinstance(model, DataParallel):
        model_core = model.module
    else:
        model_core = model

    # 2. Loop through query images
    print(f"Processing {len(query_imgs_pil)} query images...")
    
    for q_idx, query_img_pil in enumerate(query_imgs_pil):
        # Preprocess query
        query_image_tensor = T.ToTensor()(query_img_pil)
        dummy_bboxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
        query_img_resized, _, query_scale = resize_and_pad(
            query_image_tensor, dummy_bboxes, full_stretch=False
        )
        query_img_norm = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(query_img_resized).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs, _, _, _, masks = model_core.forward_cross(
                supp_img_norm, supp_bboxes_norm, query_img_norm
            )

        # Post-processing
        idx = 0
        thr = 4
        
        # Filter by threshold
        score_mask = outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr
        pred_boxes = outputs[idx]['pred_boxes'][score_mask]
        pred_scores = outputs[idx]['box_v'][score_mask]
        
        keep = ops.nms(pred_boxes, pred_scores, 0.5)
        
        boxes = pred_boxes[keep]
        scores = pred_scores[keep]
        
        # Clamp and restore coordinates
        final_bboxes = torch.clamp(boxes, 0, 1)
        
        padded_h, padded_w = query_img_resized.shape[1], query_img_resized.shape[2]
        
        final_bboxes[:, 0] *= padded_w
        final_bboxes[:, 1] *= padded_h
        final_bboxes[:, 2] *= padded_w
        final_bboxes[:, 3] *= padded_h
        
        final_bboxes /= query_scale
        
        # Combine boxes and scores: [x1, y1, x2, y2, score]
        final_results = torch.cat([final_bboxes, scores.unsqueeze(1)], dim=1)
        
        boxes_list = final_results.cpu().numpy().tolist()
        results.append(boxes_list)
        
    return results

class Config:
    def __init__(self):
        self.reduction = 16
        self.image_size = 1024
        self.emb_dim = 256
        self.num_heads = 8
        self.kernel_dim = 1
        self.num_objects = 3
        self.backbone_lr = 0
        self.zero_shot = False
        self.output_masks = False
        self.model_path = ""

def load_model(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    args = Config()
    args.model_path = model_path
    
    model = build_model(args)
    model = model.to(device)
    
    if torch.cuda.is_available():
        model = DataParallel(model)
        
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        raise

    model.eval()
    return model

def run_pipeline(support_img, support_box, query_imgs, model_path="GeCo.pth"):
    """
    End-to-end pipeline: Load model -> Inference -> Return results
    
    Args:
        support_img: PIL Image or path
        support_box: [x1, y1, x2, y2]
        query_imgs: List of (PIL Image or path)
        model_path: Path to .pth file
        
    Returns:
        results: List of lists of boxes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(model_path, device)
    
    # Handle paths if strings are passed
    if isinstance(support_img, str):
        support_img = Image.open(support_img).convert("RGB")
        
    query_imgs_pil = []
    for q in query_imgs:
        if isinstance(q, str):
            query_imgs_pil.append(Image.open(q).convert("RGB"))
        else:
            query_imgs_pil.append(q)
            
    # Run inference
    return run_batch_inference(model, support_img, support_box, query_imgs_pil, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('GeCo Cross-Image Demo', parents=[get_argparser()])
    parser.add_argument('--support_image_path', default=None, help='Path to support image (optional; will open file dialog if omitted)')
    parser.add_argument('--query_image_path', default=None, help='Path to query image (optional; will open file dialog if omitted)')
    args = parser.parse_args()
    demo_cross(args)
