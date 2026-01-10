from torch.nn import DataParallel

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import ops
from torchvision import transforms as T

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from models.eefficientvitsam_geco_infer import build_model
from utils.arg_parser import get_argparser
from utils.data import resize_and_pad

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


def on_press(event):
    global start_x, start_y, rect
    if event.inaxes:
        start_x, start_y = event.xdata, event.ydata
        rect = patches.Rectangle((start_x, start_y), 0, 0, linewidth=2, edgecolor='r', facecolor='none')
        event.inaxes.add_patch(rect)
        plt.draw()


def on_motion(event):
    global start_x, start_y, rect
    if rect is not None and event.inaxes:
        width = event.xdata - start_x
        height = event.ydata - start_y
        rect.set_width(width)
        rect.set_height(height)
        plt.draw()


def on_release(event):
    global rect
    if rect is not None:
        bounding_boxes.append([
            rect.get_x(),
            rect.get_y(),
            rect.get_x() + rect.get_width(),
            rect.get_y() + rect.get_height()
        ])
        rect = None


def _choose_image_file(title: str):
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
def demo(args):
    bounding_boxes.clear()

    img_path = getattr(args, 'image_path', None)
    image_path_explicit = any(a.startswith('--image_path') for a in sys.argv[1:])

    if (not image_path_explicit) or (not img_path) or (not os.path.exists(img_path)):
        picked = _choose_image_file("Select image")
        if picked:
            img_path = picked

    if not img_path or not os.path.exists(img_path):
        print("No image selected. Exiting.")
        return

    gpu = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device = torch.device(gpu)
    else:
        device = torch.device("cpu")

    model = DataParallel(
        build_model(args).to(device),
        device_ids=[gpu] if torch.cuda.is_available() else None,
        output_device=gpu if torch.cuda.is_available() else None,
    )

    # Load GeCo weights
    try:
        checkpoint = torch.load('GeCo.pth', map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    except FileNotFoundError:
        print("Error: GeCo.pth not found. Please ensure the model weights are in the current directory.")
        return

    model.eval()

    image = T.ToTensor()(Image.open(img_path).convert("RGB"))

    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0))
    plt.axis('off')

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    plt.title("Click and drag to draw bboxes, then close window")
    plt.show()

    if not bounding_boxes:
        print("No bounding box drawn. Exiting.")
        return

    bboxes = torch.tensor(bounding_boxes, dtype=torch.float32)

    img, bboxes_resized, scale = resize_and_pad(image, bboxes, full_stretch=False)
    img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0).to(device)
    bboxes_in = bboxes_resized.unsqueeze(0).to(device)

    outputs, _, _, _, masks = model(img, bboxes_in)
    idx = 0
    thr = 4

    scores = outputs[idx].get('box_v', None)
    pred_boxes_raw = outputs[idx].get('pred_boxes', None)
    if scores is None or pred_boxes_raw is None:
        print("Model output missing 'box_v' or 'pred_boxes'.")
        return

    # Normalize shapes to (N,) and (N, 4)
    if scores.dim() == 2 and scores.size(0) == 1:
        scores = scores[0]
    if pred_boxes_raw.dim() == 3 and pred_boxes_raw.size(0) == 1:
        pred_boxes_raw = pred_boxes_raw[0]

    if scores.numel() == 0 or pred_boxes_raw.numel() == 0:
        print("No predicted boxes (empty outputs). Try another image or check weights/backbone.")
        return

    max_score = scores.max()
    score_mask = scores > (max_score / thr)
    if score_mask.sum().item() == 0:
        print("No boxes pass the score threshold. Try lowering `thr`.")
        return

    boxes_for_nms = pred_boxes_raw[score_mask]
    scores_for_nms = scores[score_mask]
    if boxes_for_nms.numel() == 0:
        print("No boxes available after filtering.")
        return

    keep = ops.nms(boxes_for_nms, scores_for_nms, 0.5)
    boxes = boxes_for_nms[keep]

    bboxes_pred = torch.clamp(boxes, 0, 1)

    plt.clf()
    plt.imshow(image.permute(1, 2, 0))

    pred_boxes = bboxes_pred.cpu() / torch.tensor([scale, scale, scale, scale]) * img.shape[-1]
    for box in pred_boxes:
        box = box.tolist()
        plt.plot([box[0], box[0], box[2], box[2], box[0]], [box[1], box[3], box[3], box[1], box[1]], linewidth=0.7, color='orange')

    for box in bounding_boxes:
        plt.plot([box[0], box[0], box[2], box[2], box[0]], [box[1], box[3], box[3], box[1], box[1]], linewidth=2, color='red')

    plt.title("Number of selected objects:" + str(len(bboxes_pred)))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GeCo (EfficientViT-SAM backbone)', parents=[get_argparser()])

    # Optional, used by models/eefficientvitsam_geco_infer.py if present.
    parser.add_argument('--efficientvit_sam_ckpt', default=None, help='Path to EfficientViT-SAM checkpoint (optional)')
    parser.add_argument('--efficientvit_sam_type', default=None, help='EfficientViT-SAM model type/name (optional)')

    # Optional: SAM-H weights for refinement stage (mask_decoder/prompt_encoder).
    parser.add_argument('--sam_refine_ckpt', default='sam_vit_h_4b8939.pth', help='Path to SAM ViT-H checkpoint for refinement (optional)')
    parser.add_argument('--disable_sam_refine', action='store_true', help='Disable SAM-based refinement (coarser boxes, but no SAM checkpoint needed)')

    args = parser.parse_args()
    demo(args)
