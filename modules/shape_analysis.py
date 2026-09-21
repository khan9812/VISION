"""
Shape Analysis Module for Particle Analysis
===========================================
CLIP-based morphological classification and visualization.

Functions:
- classify_shapes_with_clip: CLIP-based shape classification
- generate_shape_colors: Automatic color assignment for shapes
- get_best_shape_examples: Extract representative examples
- calculate_shape_metrics: Compute shape distribution metrics
- create_shape_pie_chart: Pie chart visualization
- create_shape_overlay: Shape-colored overlay visualization
- create_shape_examples_gallery: Example particles gallery
- create_confidence_boxplot: Confidence distribution visualization
"""

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
from scipy.stats import entropy as scipy_entropy
from modules.color_palette import PALETTE_NEUTRAL, palette_hex, palette_rgb
from modules.scientific_plotting import (
    create_shape_composition_figure,
    shape_color_sequence,
)


def crop_and_zoom_particle(
    mask_image,
    segmentation_mask=None,
    padding_ratio=0.15,
    target_size=224,
):
    """
    Crop particle from mask and zoom to fill frame for better CLIP classification.

    Args:
        mask_image: np.ndarray
            Particle image with black background
        padding_ratio: float
            Proportion of bounding box to add as padding (0.15 = 15%)
        target_size: int
            Final square image size (224 for CLIP)

    Returns:
        PIL.Image: Cropped and zoomed particle
    """
    # Convert to grayscale if needed
    if len(mask_image.shape) == 3:
        gray = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = mask_image.copy()

    # Find the particle bounding box from its segmentation mask. Falling back
    # to intensity is retained only for older callers without a mask.
    if segmentation_mask is not None:
        coords = np.column_stack(np.where(np.asarray(segmentation_mask, dtype=bool)))
    else:
        coords = np.column_stack(np.where(gray > 0))
    if len(coords) == 0:
        # No particle found, return white image
        return Image.fromarray(np.ones((target_size, target_size), dtype=np.uint8) * 255)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Calculate padding
    height = y_max - y_min + 1
    width = x_max - x_min + 1
    pad_h = int(height * padding_ratio)
    pad_w = int(width * padding_ratio)

    # Apply padding with boundary checks
    y_min = max(0, y_min - pad_h)
    y_max = min(mask_image.shape[0], y_max + 1 + pad_h)
    x_min = max(0, x_min - pad_w)
    x_max = min(mask_image.shape[1], x_max + 1 + pad_w)

    # Crop particle region
    if len(mask_image.shape) == 3:
        cropped = mask_image[y_min:y_max, x_min:x_max, :]
    else:
        cropped = mask_image[y_min:y_max, x_min:x_max]

    # Create square canvas with white background
    crop_h, crop_w = cropped.shape[:2]
    max_dim = max(crop_h, crop_w)

    if len(mask_image.shape) == 3:
        square_canvas = np.ones((max_dim, max_dim, mask_image.shape[2]), dtype=np.uint8) * 255
        # Center particle in canvas
        y_offset = (max_dim - crop_h) // 2
        x_offset = (max_dim - crop_w) // 2
        square_canvas[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w, :] = cropped
    else:
        square_canvas = np.ones((max_dim, max_dim), dtype=np.uint8) * 255
        y_offset = (max_dim - crop_h) // 2
        x_offset = (max_dim - crop_w) // 2
        square_canvas[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = cropped

    # Resize to target size
    resized = cv2.resize(square_canvas, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

    # Convert to RGB if grayscale
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(resized)


def classify_shapes_with_clip(masks, original_image, shape_labels, shape_descriptions,
                              model, preprocess, text_tokens, device, batch_size=64,
                              temperature=None, confidence_threshold=None):
    """
    Classify particle shapes using CLIP model.

    Args:
        masks: list
            List of mask dictionaries with 'segmentation' key
        original_image: np.ndarray
            Original particle image
        shape_labels: list[str]
            Shape category names (e.g., ["Square", "Circle", "Triangle"])
        shape_descriptions: list[str] or list[list[str]]
            One prompt per class, or a prompt ensemble grouped by class.
        model: CLIP model
            Loaded CLIP model
        preprocess: torchvision.transforms
            CLIP preprocessing pipeline
        text_tokens: torch.Tensor
            Pre-tokenized prompts. For an ensemble, prompts must be flattened in
            the same class order as shape_descriptions.
        device: str
            "cuda" or "cpu"
        batch_size: int
            Batch size for inference
        temperature: float or None
            CLIP temperature parameter. If None, uses model's learned logit_scale.
        confidence_threshold: float or None
            Minimum confidence to assign label. If None, always return highest probability shape.

    Returns:
        tuple: (shapes, confidences, shape_counts)
            - shapes: list[str] - Assigned shape for each particle
            - confidences: list[float] - Confidence scores
            - shape_counts: Counter - Shape distribution
    """
    # Use model's learned logit_scale if temperature not specified
    if temperature is None:
        logit_scale = model.logit_scale.exp().item()
    else:
        logit_scale = 1.0 / temperature

    print(f"\n[INFO] Classifying {len(masks)} particles with CLIP...")
    print(f"   Shape categories: {', '.join(shape_labels)}")
    print(f"   Logit scale: {logit_scale:.2f} (learned)" if temperature is None else f"   Temperature: {temperature}")
    print(f"   Confidence threshold: {confidence_threshold if confidence_threshold else 'None (always highest prob)'}")

    # Handle empty masks case
    if len(masks) == 0:
        print("   [INFO] No masks to classify - returning empty results")
        return [], [], Counter()

    is_prompt_ensemble = (
        len(shape_descriptions) == len(shape_labels)
        and all(isinstance(prompts, (list, tuple)) for prompts in shape_descriptions)
    )
    if is_prompt_ensemble:
        prompt_counts = [len(prompts) for prompts in shape_descriptions]
        if any(count == 0 for count in prompt_counts):
            raise ValueError("Each CLIP class must have at least one prompt")
    else:
        prompt_counts = [1] * len(shape_labels)

    if sum(prompt_counts) != int(text_tokens.shape[0]):
        raise ValueError(
            "CLIP prompt count does not match text tokens: "
            f"expected {sum(prompt_counts)}, received {int(text_tokens.shape[0])}"
        )

    with torch.no_grad():
        prompt_features = model.encode_text(text_tokens).float()
        prompt_features /= prompt_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        class_text_features = []
        prompt_offset = 0
        for prompt_count in prompt_counts:
            class_feature = prompt_features[
                prompt_offset:prompt_offset + prompt_count
            ].mean(dim=0)
            class_feature /= class_feature.norm().clamp_min(1e-12)
            class_text_features.append(class_feature)
            prompt_offset += prompt_count
        text_features = torch.stack(class_text_features, dim=0)

    if is_prompt_ensemble:
        prompt_summary = ", ".join(
            f"{label}={count}" for label, count in zip(shape_labels, prompt_counts)
        )
        print(f"   Prompt ensemble: {prompt_summary}")

    # Extract and preprocess particles
    preprocessed_images = []
    for mask in masks:
        # Extract mask region
        segmentation = mask['segmentation']
        mask_image = np.zeros_like(original_image, dtype=np.uint8)
        mask_image[segmentation] = original_image[segmentation]

        # Images enter VISION through OpenCV (BGR), whereas CLIP/PIL expects
        # RGB. Grayscale EM images are unchanged; color images are now correct.
        if mask_image.ndim == 3 and mask_image.shape[2] == 3:
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

        # Ensure uint8
        if mask_image.max() <= 1.0:
            mask_image = (mask_image * 255).astype(np.uint8)
        elif mask_image.dtype != np.uint8:
            mask_image = mask_image.astype(np.uint8)

        # Crop and zoom
        pil_image = crop_and_zoom_particle(
            mask_image,
            segmentation_mask=segmentation,
            padding_ratio=0.15,
            target_size=224,
        )
        preprocessed_images.append(preprocess(pil_image).unsqueeze(0))

    preprocessed_images = torch.cat(preprocessed_images, dim=0).to(device)

    # Classification
    shapes = []
    confidences = []

    for i in range(0, len(preprocessed_images), batch_size):
        image_batch = preprocessed_images[i:i + batch_size]

        with torch.no_grad():
            image_features = model.encode_image(image_batch).float()

            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            # Compute similarity using learned logit_scale
            similarity = logit_scale * (image_features @ text_features.T)
            probs = similarity.softmax(dim=-1)

            # Process each image individually with scaled confidence
            for prob in probs.cpu():
                top_prob, top_label = prob.max(dim=-1)

                # Confidence scaling (Z-score normalization)
                mean_confidence = prob.mean().item()
                std_confidence = prob.std().item()

                if std_confidence > 0:
                    scaled_confidence = (top_prob.item() - mean_confidence) / std_confidence
                else:
                    scaled_confidence = top_prob.item()

                # Always return the highest probability label (no threshold check)
                # confidence_threshold is ignored - always use top prediction
                shapes.append(shape_labels[top_label.item()])
                confidences.append(float(top_prob.item()))

    # Count shapes
    shape_counts = Counter(shapes)

    print("[OK] Classification complete!")
    print(f"   Shape distribution:")
    for shape, count in shape_counts.most_common():
        percentage = (count / len(shapes)) * 100
        print(f"      {shape}: {count} ({percentage:.1f}%)")

    return shapes, confidences, shape_counts

def generate_shape_colors(unique_shapes):
    """
    Generate distinct colors for each shape using the project palette.

    Args:
        unique_shapes: list[str]
            List of unique shape names

    Returns:
        dict: Mapping from shape name to RGB color tuple
    """
    n_shapes = len(unique_shapes)
    colors = shape_color_sequence(n_shapes)
    shape_to_color = {shape: colors[i] for i, shape in enumerate(unique_shapes)}
    return shape_to_color


def get_best_shape_examples(shapes, confidences, masks, confidence_threshold=0.9):
    """
    Extract one best example per shape (highest confidence > threshold).

    Args:
        shapes: list[str]
            Shape labels for each particle
        confidences: list[float]
            Confidence scores
        masks: list
            Particle masks
        confidence_threshold: float
            Minimum confidence to include example

    Returns:
        dict: {shape_name: {'index': int, 'confidence': float, 'mask': dict}}
    """
    unique_shapes = list(set(shapes))
    shape_examples = {}

    for shape in unique_shapes:
        # Find all particles of this shape
        indices = [i for i, s in enumerate(shapes) if s == shape]

        # Filter by confidence threshold
        high_conf = [(i, confidences[i]) for i in indices if confidences[i] > confidence_threshold]

        if high_conf:
            # Select highest confidence
            best_idx, best_conf = max(high_conf, key=lambda x: x[1])
            shape_examples[shape] = {
                'index': best_idx,
                'confidence': best_conf,
                'mask': masks[best_idx]
            }

    return shape_examples


def calculate_shape_metrics(shapes, confidences):
    """
    Calculate comprehensive shape distribution metrics.

    Metrics:
    - Percentages: Distribution of each shape
    - Shannon Entropy: Diversity measure
    - Simpson Diversity: 1 - sum(p_i^2)
    - Dominant Shape: Most common shape
    - Average Confidence: Mean CLIP confidence

    Args:
        shapes: list[str]
            Shape labels
        confidences: list[float]
            Confidence scores

    Returns:
        dict: Shape metrics
    """
    shape_counts = Counter(shapes)
    total = len(shapes)

    # Percentages
    percentages = {shape: (count / total) * 100 for shape, count in shape_counts.items()}

    # Probabilities for diversity metrics
    probs = np.array([count / total for count in shape_counts.values()])

    # Shannon Entropy: H = -sum(p_i * log(p_i))
    shannon_entropy = scipy_entropy(probs, base=2)  # bits

    # Simpson Diversity: D = 1 - sum(p_i^2)
    simpson_diversity = 1 - np.sum(probs ** 2)

    # Dominant shape
    dominant_shape = shape_counts.most_common(1)[0][0]
    dominant_percentage = percentages[dominant_shape]

    # Average confidence
    avg_confidence = np.mean(confidences)

    metrics = {
        'total_particles': total,
        'shape_counts': dict(shape_counts),
        'percentages': percentages,
        'shannon_entropy': shannon_entropy,
        'simpson_diversity': simpson_diversity,
        'dominant_shape': dominant_shape,
        'dominant_percentage': dominant_percentage,
        'avg_confidence': avg_confidence
    }

    return metrics


def create_shape_pie_chart(shape_counts, color_map, figsize=(8, 8)):
    """Create the shared publication-style shape composition donut chart."""
    return create_shape_composition_figure(shape_counts, color_map=color_map, figsize=figsize)


def create_shape_overlay(image, masks, shapes, color_map, figsize=(10, 10)):
    """
    Create shape-colored overlay visualization.

    Args:
        image: np.ndarray
            Original image
        masks: list
            Particle masks
        shapes: list[str]
            Shape labels
        color_map: dict
            Shape to color mapping
        figsize: tuple
            Figure size

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Display base image
    if len(image.shape) == 3:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(image, cmap='gray')

    # Overlay shapes with colors
    for mask, shape in zip(masks, shapes):
        segmentation = mask['segmentation']
        color = color_map.get(shape, palette_rgb(0))  # Gray for unknown

        # Create colored mask overlay
        overlay = np.zeros((*segmentation.shape, 4))
        overlay[segmentation] = [*color, 0.5]  # RGBA with alpha=0.5

        ax.imshow(overlay)

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=shape)
                      for shape, color in color_map.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_title('Projected-morphology classification', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    return fig


def create_shape_examples_gallery(shape_examples, original_image, color_map, figsize=(12, 8)):
    """
    Create gallery of representative shape examples (one per shape).

    Args:
        shape_examples: dict
            Output from get_best_shape_examples()
        original_image: np.ndarray
            Original particle image
        color_map: dict
            Shape to color mapping
        figsize: tuple
            Figure size

    Returns:
        matplotlib.figure.Figure
    """
    n_shapes = len(shape_examples)

    if n_shapes == 0:
        # No examples found
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No high-confidence examples found\n(confidence > 0.9)',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Create grid
    cols = min(4, n_shapes)
    rows = (n_shapes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot examples
    for idx, (shape, example) in enumerate(sorted(shape_examples.items())):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # Extract particle
        mask = example['mask']['segmentation']
        mask_image = np.zeros_like(original_image)
        mask_image[mask] = original_image[mask]

        # Display
        if len(mask_image.shape) == 3:
            ax.imshow(cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(mask_image, cmap='gray')

        # Title with shape and confidence
        color = color_map.get(shape, palette_rgb(0))
        ax.set_title(f"{shape}\n(conf: {example['confidence']:.3f})",
                     fontsize=11, color=color, fontweight='bold')
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n_shapes, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    fig.suptitle('Representative projected morphologies (best confidence > 0.9)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def create_confidence_boxplot(shapes, confidences, color_map, figsize=(10, 6)):
    """
    Create box plot of CLIP confidence distribution by shape.

    Args:
        shapes: list[str]
            Shape labels
        confidences: list[float]
            Confidence scores
        color_map: dict
            Shape to color mapping
        figsize: tuple
            Figure size

    Returns:
        matplotlib.figure.Figure
    """
    # Prepare data
    unique_shapes = sorted(set(shapes))
    data = []
    colors = []

    for shape in unique_shapes:
        shape_conf = [confidences[i] for i, s in enumerate(shapes) if s == shape]
        data.append(shape_conf)
        colors.append(color_map.get(shape, palette_rgb(0)))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Box plot
    bp = ax.boxplot(data, labels=unique_shapes, patch_artist=True,
                    boxprops=dict(facecolor=palette_hex(1)),
                    medianprops=dict(color=palette_hex(0), linewidth=2))

    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel('Morphology category', fontsize=12, fontweight='bold')
    ax.set_ylabel('CLIP Confidence Score', fontsize=12, fontweight='bold')
    ax.set_title('Projected-morphology classification confidence',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # Rotate x labels if many shapes
    if len(unique_shapes) > 5:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test module
    print("Shape Analysis Module")
    print("=" * 60)
    print("\nAvailable functions:")
    print("  - classify_shapes_with_clip(...)")
    print("  - generate_shape_colors(unique_shapes)")
    print("  - get_best_shape_examples(shapes, confidences, masks, threshold)")
    print("  - calculate_shape_metrics(shapes, confidences)")
    print("  - create_shape_pie_chart(shape_counts, color_map, figsize)")
    print("  - create_shape_overlay(image, masks, shapes, color_map, figsize)")
    print("  - create_shape_examples_gallery(examples, image, color_map, figsize)")
    print("  - create_confidence_boxplot(shapes, confidences, color_map, figsize)")
