import os

import matplotlib as mpl
from torchvision import datasets


def apply_dlai_style():
    # Global plot style
    PLOT_STYLE = {
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "font.family": "sans",  # "sans-serif",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 3,
        "lines.markersize": 6,
    }

    # mpl.rcParams.update(PLOT_STYLE)

    # Custom colors (reusable)
    color_map = {
        "pink": "#F65B66",
        "blue": "#1C74EB",
        "yellow": "#FAB901",
        "red": "#DD3C66",
        "purple": "#A12F9D",
        "cyan": "#237B94",
    }
    return color_map, PLOT_STYLE


color_map, PLOT_STYLE = apply_dlai_style()
mpl.rcParams.update(PLOT_STYLE)


def get_dataset():
    # Define the path to the dataset
    path_dataset = "./dataset"

    # If the dataset path does not exist, create it
    if not os.path.exists(path_dataset):
        os.makedirs(path_dataset)

        # Download the Fashion MNIST validation dataset (from pyTorch)
        datasets.FashionMNIST(path_dataset, train=False, download=True)

    else:
        print("Dataset already exists.")
    # if the dataset is not dowloaded, download fashion_mnist

    # Load the Fashion MNIST validation dataset (from pyTorch)
    dataset = datasets.FashionMNIST(path_dataset, train=False, download=False)

    return dataset


def plot_counting(counting_params):
    import matplotlib.pyplot as plt

    # Plotting
    plt.figure(figsize=(14, 8))
    plt.bar(counting_params.keys(), counting_params.values())
    plt.xlabel("Layer Name")
    plt.ylabel("Number of Parameters")
    plt.title("Number of Parameters in Each Terminal Layer")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
