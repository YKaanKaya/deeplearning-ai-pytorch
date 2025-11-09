import os
import sys

import matplotlib.pyplot as plt
from directory_tree import DisplayTree
from fastai.vision.all import show_image, show_titled_image
from tqdm.auto import tqdm



def get_dataloader_bar(dataloader, color="green"):
    """
    Generates and returns a tqdm progress bar for a dataloader.

    Args:
        dataloader: The data loader for which to create the progress bar.
        color (str): The color of the progress bar.
    """
    # Get the total number of samples from the dataloader's dataset.
    num_samples = len(dataloader.dataset)

    # Initialize a tqdm progress bar with specified settings.
    pbar = tqdm(
        # Set the total number of iterations for the bar.
        total=num_samples,
        # Dynamically calculate the width of the progress bar.
        ncols=int(num_samples / 10) + 300,
        # Define the format string for the progress bar's appearance.
        bar_format="{desc} {bar} {postfix}",
        # Direct the progress bar output to the standard output stream.
        file=sys.stdout,
        # Set the color of the progress bar.
        colour=color,
    )

    # Return the configured progress bar object.
    return pbar



def update_dataloader_bar(p_bar, batch, current_bs, n_samples):
    """
    Updates a given tqdm progress bar with the current batch processing status.

    Args:
        p_bar: The tqdm progress bar object to update.
        batch (int): The current batch index (zero-based).
        current_bs (int): The size of the current batch.
        n_samples (int): The total number of samples in the dataset.
    """
    # Advance the progress bar by the number of items in the current batch.
    p_bar.update(current_bs)
    # Set the description to show the current batch number.
    p_bar.set_description(f"Batch {batch+1}")

    # Check if the current batch is the last one.
    if (batch + 1) * current_bs > n_samples:
        # Update the postfix to show the total number of samples processed.
        p_bar.set_postfix_str(f"{n_samples} of a total of  {n_samples} samples")
    else:
        # Update the postfix to show the cumulative number of samples processed.
        p_bar.set_postfix_str(
            f"{current_bs*(batch+1)} of a total of  {n_samples} samples"
        )



def plot_img(img, label=None, info=None, ax=None):
    """
    Displays an image with an optional label and supplementary information.

    Args:
        img: The image to be displayed.
        label: An optional label to be used as the image title.
        info: Optional text to display below the image.
        ax: An optional matplotlib axes object to plot on. If not provided,
            a new figure and axes will be created.
    """

    def add_info_text(ax, info):
        """
        Adds supplementary text below the plot on a given axes.

        Args:
            ax: The matplotlib axes object.
            info (str): The text to be added.
        """
        # Add text to the axes at a specified position.
        ax.text(
            0.5, -0.1, info, transform=ax.transAxes, ha="center", va="top", fontsize=10
        )
        # Set the x-axis label position to the top.
        ax.xaxis.set_label_position("top")

    # Create a new figure and axes if none are provided.
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Check if a label is provided to determine how to display the image.
    if label:
        # Create a title string with the provided label.
        title = f"Label: {label}"
        # Display the image with the generated title.
        show_titled_image((img, title), ax=ax)
    else:
        # Display the image without a title.
        show_image(img, ax=ax)

    # Check if supplementary information is provided.
    if info:
        # Add the information as text below the image.
        add_info_text(ax, info)

    # If no axes were passed in, display the newly created plot.
    if ax is None:
        plt.show()



def get_grid(num_rows, num_cols, figsize=(16, 8)):
    """
    Creates a grid of subplots within a Matplotlib figure.

    This function handles cases where the grid has only one row or one
    column to ensure the returned axes object is always a 2D iterable.

    Args:
        num_rows (int): The number of rows in the subplot grid.
        num_cols (int): The number of columns in the subplot grid.
        figsize (tuple): The width and height of the figure in inches.

    Returns:
        tuple: A tuple containing the Matplotlib figure and a 2D list of axes objects.
    """
    # Create a figure and a set of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Handle the case where there is only one row.
    if num_rows == 1:
        # Ensure the axes object is iterable for consistency.
        axes = [axes]
    # Handle the case where there is only one column.
    elif num_cols == 1:
        # Ensure the axes object is a 2D list for consistent indexing.
        axes = [[ax] for ax in axes]
        
    # Return the figure and the formatted axes grid.
    return fig, axes



def print_data_folder_structure(root_dir, max_depth=1):
    """
    Displays the folder and file structure of a given directory.

    Args:
        root_dir (str): The path to the root directory to be displayed.
        max_depth (int): The maximum depth to traverse the directory tree.
    """
    # Define the configuration settings for displaying the directory tree.
    config_tree = {
        # Specify the starting path for the directory tree.
        "dirPath": root_dir,
        # Set to False to include both files and directories.
        "onlyDirs": False,
        # Set the maximum depth for the tree traversal.
        "maxDepth": max_depth,
        # Specify a sorting option (100 typically means no specific sort).
        "sortBy": 100,
    }
    # Create and display the tree structure using the unpacked configuration.
    DisplayTree(**config_tree)



def explore_extensions(root_dir):
    """
    Scans a directory and its subdirectories to catalog files by their extension.

    Args:
        root_dir (str): The path to the root directory to scan.

    Returns:
        dict: A dictionary where keys are the unique file extensions found (in lowercase)
              and values are lists of full file paths for each extension.
    """
    # Initialize a dictionary to store file paths, grouped by extension.
    extensions = {}
    # Walk through the directory tree starting from the root directory.
    for dirpath, _, filenames in os.walk(root_dir):
        # Iterate over each file in the current directory.
        for filename in filenames:
            # Extract the file extension and convert it to lowercase.
            ext = os.path.splitext(filename)[1].lower()
            # If the extension has not been seen before, add it to the dictionary.
            if ext not in extensions:
                # Initialize a new list for this extension.
                extensions[ext] = []
            # Append the full path of the file to the list for its extension.
            extensions[ext].append(os.path.join(dirpath, filename))
    # Return the dictionary of extensions and their corresponding file paths.
    return extensions



def quick_debug(img):
    """
    Prints key debugging information about an image tensor.

    This function displays the shape, data type, and value range of a given
    image tensor to help with quick diagnostics.

    Args:
        img: The image tensor to inspect.
    """
    # Print the shape of the image tensor.
    print(f"Shape: {img.shape}")  # Should be [3, 224, 224]
    # Print the data type of the tensor.
    print(f"Type: {img.dtype}")  # Should be torch.float32
    # Print the minimum and maximum pixel values in the tensor.
    print(
        f"Range of pixel values: [{img.min():.1f}, {img.max():.1f}]"
    )  # Should be around [-2, 2]# Should be around [-2, 2]


# def get_mean_std():
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     return mean, std
