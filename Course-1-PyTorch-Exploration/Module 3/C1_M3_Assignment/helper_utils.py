import os

import matplotlib.pyplot as plt
import numpy as np
from directory_tree import DisplayTree
from fastai.vision.core import show_image, show_titled_image
from torchvision import transforms


class Denormalize:
    def __init__(self, mean, std):

        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]

        self.denormalize = transforms.Normalize(mean=new_mean, std=new_std)

    def __call__(self, tensor):
        return self.denormalize(tensor)


def plot_img(img, label=None, info=None, ax=None):

    def add_info_text(ax, info):
        ax.text(
            0.5, -0.1, info, transform=ax.transAxes, ha="center", va="top", fontsize=10
        )
        ax.xaxis.set_label_position("top")

    # using show_image from fastai to handle different image types
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    if label:
        title = f"Label: {label}"
        show_titled_image((img, title), ax=ax)
    else:
        show_image(img, ax=ax)

    if info:
        # Add info as text below the image
        add_info_text(ax, info)

    if ax is None:
        plt.show()


def get_grid(num_rows, num_cols, figsize=(16, 8)):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_rows == 1:
        axes = [axes]  # Ensure axes is iterable
    elif num_cols == 1:
        axes = [[ax] for ax in axes]  # Ensure 2D list
    return fig, axes


def print_data_folder_structure(root_dir, max_depth=1):
    """Print the folder structure of the dataset directory."""
    config_tree = {
        "dirPath": root_dir,
        "onlyDirs": False,
        "maxDepth": max_depth,
        "sortBy": 1,  # Sort by type (files first, then folders)
    }
    DisplayTree(**config_tree)


def explore_extensions(root_dir):
    """Explore and print the file extensions in the dataset directory."""
    extensions = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in extensions:
                extensions[ext] = []
            extensions[ext].append(os.path.join(dirpath, filename))
    return extensions


def visual_exploration(dataset, num_rows=2, num_cols=4):
    """Visual exploration of the dataset by displaying random samples in a grid."""
    # Calculate total number of samples to display
    total_samples = num_rows * num_cols

    # Randomly select indices from the dataset without replacement
    # This ensures we get a diverse sample of the dataset
    indices = np.random.choice(len(dataset), total_samples, replace=False)

    # Create a grid of subplots with appropriate figure size
    # Each subplot gets (3 x 4) inches per image for good visibility
    fig, axes = get_grid(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 4))

    # Iterate through each axis and corresponding random index
    for ax, idx in zip(axes.flatten(), indices):
        # Load image and label from dataset at the random index
        image, label = dataset[idx]

        # Get human-readable description for the label
        description = dataset.get_label_description(label)

        # Create a combined label string with both number and description
        label = f"{label} - {description}"

        # Create info string showing index and image dimensions
        info = f"Index: {idx} Size: {image.size}"

        # Plot the image on the current axis with label and info
        plot_img(image, label=label, info=info, ax=ax)

    # Display the complete grid of images
    plt.show()
