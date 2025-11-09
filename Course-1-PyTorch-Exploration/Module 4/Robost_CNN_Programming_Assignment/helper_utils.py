import itertools
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, TensorDataset



def load_cifar100_subset(target_classes, train_transform, val_transform, root='./cifar_100'):
    """
    Loads and filters the CIFAR-100 dataset to include only specified target classes.

    This function first checks for a local copy of the CIFAR-100 dataset and
    downloads it if not found. It then filters both the training and test sets
    to retain only the images and labels corresponding to the classes specified
    in `target_classes`. The labels are remapped to be contiguous from 0.

    Args:
        target_classes: A list of class name strings to be included in the dataset subset.
        train_transform: A torchvision transform to be applied to the training dataset images.
        val_transform: A torchvision transform to be applied to the test/val dataset images.
        root: The root directory where the dataset is stored or will be downloaded.

    Returns:
        A tuple containing the filtered training dataset and the filtered test dataset.
        Returns (None, None) if a specified target class is not found.
    """
    # Construct the path to the CIFAR-100 dataset directory.
    cifar100_path = os.path.join(root, 'cifar-100-python')
    # Check if the dataset directory exists locally.
    if os.path.isdir(cifar100_path):
        print(f"Dataset found in '{root}'. Loading from local files.")
    # If not found, inform the user that it will be downloaded.
    else:
        print(f"Dataset not found in '{root}'. Downloading...")

    # Load the full CIFAR-100 training dataset.
    train_dataset_full = torchvision.datasets.CIFAR100(
        root=root, 
        train=True, 
        download=True, 
        transform=train_transform
    )

    # Load the full CIFAR-100 test dataset.
    test_dataset_full = torchvision.datasets.CIFAR100(
        root=root, 
        train=False, 
        download=True, 
        transform=val_transform
    )
    print("Dataset loaded successfully.")

    # Get the list of all class names from the dataset.
    all_classes = train_dataset_full.classes
    try:
        # Get the original integer indices for the target class names.
        target_indices = [all_classes.index(cls) for cls in target_classes]
    # Handle the case where a specified class name is not in the dataset.
    except ValueError as e:
        print(f"Error: One of the target classes not found in CIFAR-100. {e}")
        return None, None
        
    # Create a mapping from the original class indices to new, contiguous indices (0, 1, 2, ...).
    label_map = {old_label: new_label for new_label, old_label in enumerate(target_indices)}

    # Define a helper function to filter a dataset based on the target classes.
    def _filter_dataset(dataset):
        # Convert the list of targets to a NumPy array for efficient boolean indexing.
        targets_np = np.array(dataset.targets)
        # Create a boolean mask to identify which samples belong to the target classes.
        indices_to_keep = np.isin(targets_np, target_indices)
        
        # Filter the dataset's image data using the boolean mask.
        dataset.data = dataset.data[indices_to_keep]
        
        # Get the original labels of the samples that are being kept.
        original_targets_to_keep = targets_np[indices_to_keep]
        # Remap the original labels to the new contiguous labels.
        dataset.targets = [label_map[target] for target in original_targets_to_keep]
        
        # Update the dataset's class list to only include the target classes.
        dataset.classes = target_classes
        return dataset

    print(f"\nFiltering for {len(target_classes)} classes...")
    # Apply the filtering logic to the full training dataset.
    train_dataset_subset = _filter_dataset(train_dataset_full)
    # Apply the filtering logic to the full test dataset.
    test_dataset_subset = _filter_dataset(test_dataset_full)
    print("Filtering complete. Returning training and validation datasets.")
    
    # Return the filtered training and test subsets.
    return train_dataset_subset, test_dataset_subset



def visualise_images(loader, grid):
    """
    Visualizes a grid of random images from a dataset, showing one image per class.

    Args:
        loader: The DataLoader object containing the dataset.
        grid: A tuple specifying the grid dimensions as (rows, cols).
    """
    # Unpack the grid dimensions from the input tuple
    rows, cols = grid
    # Calculate the total number of images to display in the grid
    num_images_to_show = rows * cols

    # Get the dataset object from the DataLoader
    dataset_to_show = loader.dataset

    # Create a dictionary to store lists of indices for each class
    class_indices = defaultdict(list)
    # Iterate through the dataset to populate the class_indices dictionary
    for idx, target in enumerate(dataset_to_show.targets):
        class_indices[target].append(idx)
        
    # Get the list of class names from the dataset
    class_names = dataset_to_show.classes

    # Create a figure and a set of subplots for the grid layout
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # Iterate over each subplot in the grid
    for i, ax in enumerate(axes.flat):
        # If the current index is out of bounds, turn off the subplot axis
        if i >= num_images_to_show or i >= len(class_names):
            ax.axis('off')
            continue
            
        # Set the class label based on the current iteration index
        class_label = i
        
        # Get the list of image indices for the current class
        indices_for_class = class_indices[class_label]
        # If there are no images for this class, turn off the subplot axis
        if not indices_for_class:
            ax.axis('off')
            continue

        # Choose a random image index from the list for the current class
        random_image_index = random.choice(indices_for_class)
        
        # Retrieve the image tensor and its corresponding label from the dataset
        image_tensor, _ = dataset_to_show[random_image_index]
        
        # Convert the tensor to a NumPy array and transpose dimensions for display
        img_to_display = image_tensor.numpy().transpose((1, 2, 0))
        
        # Rescale the pixel values from the normalized range to [0, 1] for proper visualization
        min_val = img_to_display.min()
        max_val = img_to_display.max()
        img_to_display = (img_to_display - min_val) / (max_val - min_val)
        
        # Get the name of the class corresponding to the class label
        class_name = class_names[class_label]
        
        # Display the image on the current subplot
        ax.imshow(img_to_display)
        
        # Set the title of the subplot to the capitalized class name
        ax.set_title(class_name.capitalize(), fontsize=16)
        # Turn off the axis for a cleaner look
        ax.axis('off')

    # Adjust subplot parameters for a tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Display the plot
    plt.show()
    
    

def plot_training_metrics(metrics):
    """
    Plots the training and validation metrics from a model training process.

    This function generates two side-by-side plots:
    1. Training Loss vs. Validation Loss.
    2. Validation Accuracy.

    Args:
        metrics (list): A list or tuple containing three lists:
                        [train_losses, val_losses, val_accuracies].
    """
    # Unpack the metrics into their respective lists
    train_losses, val_losses, val_accuracies = metrics
    
    # Determine the number of epochs from the length of the training losses list
    num_epochs = len(train_losses)
    # Create a 1-indexed range of epoch numbers for the x-axis
    epochs = range(1, num_epochs + 1)

    # Create a figure and a set of subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Configure the first subplot for training and validation loss ---
    # Select the first subplot
    ax1 = axes[0]
    # Plot training loss data
    ax1.plot(epochs, train_losses, color='#085c75', linewidth=2.5, marker='o', markersize=5, label='Training Loss')
    # Plot validation loss data
    ax1.plot(epochs, val_losses, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Loss')
    # Set the title and axis labels for the loss plot
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    # Display the legend
    ax1.legend()
    # Add a grid for better readability
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Configure the second subplot for validation accuracy ---
    # Select the second subplot
    ax2 = axes[1]
    # Plot validation accuracy data
    ax2.plot(epochs, val_accuracies, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Accuracy')
    # Set the title and axis labels for the accuracy plot
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    # Display the legend
    ax2.legend()
    # Add a grid for better readability
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # --- Apply dynamic and consistent styling to both subplots ---
    # Calculate a suitable interval for the x-axis ticks to avoid clutter
    x_interval = (num_epochs - 1) // 10 + 1

    # Loop through each subplot to apply common axis settings
    for ax in axes:
        # Set the y-axis to start at 0 and the x-axis to span the epochs
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=1, right=num_epochs)
        
        # Set the major tick locator for the x-axis using the dynamic interval
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_interval))
        # Set the font size for the tick labels on both axes
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Adjust subplot parameters for a tight layout
    plt.tight_layout()
    # Display the plots
    plt.show()
    
    
    
def verify_training_process(model_class, train_loader, loss_function, train_epoch_fn, device):
    """
    Verifies the training process on a small subset of data for a few epochs.

   Args:
        model_class: The model class to be instantiated for verification.
        train_loader: The DataLoader for the training dataset.
        loss_function: The loss function to be used during training.
        train_epoch_fn: The function that executes one training epoch.
        device: The device (e.g., 'cuda' or 'cpu') to run the verification on.
    """
    # Print a header for the verification process
    print("--- Verifying train_epoch (training for 5 epochs) ---\n")

    # Define the number of epochs and batches for the verification run
    NUM_VERIFY_EPOCHS = 5
    NUM_VERIFY_BATCHES = 10

    # Instantiate the model and move it to the specified device
    verify_model = model_class(15).to(device)
    # Initialize the Adam optimizer with a learning rate
    verify_optimizer = optim.Adam(verify_model.parameters(), lr=0.0005)

    # Create a small subset of the training data for quick verification
    batches = list(itertools.islice(iter(train_loader), NUM_VERIFY_BATCHES))
    # Concatenate the images and labels from the selected batches
    all_images = torch.cat([b[0] for b in batches])
    all_labels = torch.cat([b[1] for b in batches])
    # Create a TensorDataset and a DataLoader for the subset
    verify_subset_dataset = TensorDataset(all_images, all_labels)
    verify_subset_loader = DataLoader(verify_subset_dataset, batch_size=train_loader.batch_size)

    # Clone the initial weights of a specific layer to check for changes later
    initial_weight = verify_model.conv_block1.block[0].weight.clone()
    # Initialize a list to store the loss from each epoch
    epoch_losses = []

    print(f"Training on {len(verify_subset_dataset)} images for {NUM_VERIFY_EPOCHS} epochs:\n")
    # Loop through the defined number of epochs for verification
    for epoch in range(NUM_VERIFY_EPOCHS):
        # Run a single training epoch and get the loss
        loss = train_epoch_fn(
            model=verify_model,
            train_loader=verify_subset_loader,
            loss_function=loss_function,
            optimizer=verify_optimizer,
            device=device
        )
        # Append the loss to the list and print the epoch's result
        epoch_losses.append(loss)
        print(f"Epoch [{epoch+1}/{NUM_VERIFY_EPOCHS}], Loss: {loss:.4f}")

    # Get the weights of the same layer after training has completed
    trained_weight = verify_model.conv_block1.block[0].weight

    # Check if the weights have changed from their initial values
    weights_changed = not torch.equal(initial_weight, trained_weight)
    if weights_changed:
        print("\nWeight Update Check:\tModel weights changed during training.")
    else:
        print("\nWeight Update Check:\tModel weights DID NOT change.")

    # Check if the final loss is less than the initial loss
    loss_decreased = epoch_losses[-1] < epoch_losses[0]
    if loss_decreased:
        print(f"Loss Trend Check:\tLoss decreased from {epoch_losses[0]:.4f} to {epoch_losses[-1]:.4f}.")
    else:
        print(f"Loss Trend Check:\tLoss DID NOT show a decreasing trend.")
        
        
        
def verify_validation_process(model_class, val_loader, loss_function, validate_epoch_fn, device):
    """
    Verifies the validation process on a small subset of data.

    Args:
        model_class: The model class to be instantiated for verification.
        val_loader: The DataLoader for the validation dataset.
        loss_function: The loss function to be used during validation.
        validate_epoch_fn: The function that executes one validation epoch.
        device: The device (e.g., 'cuda' or 'cpu') to run the verification on.
    """
    # Print a header for the verification process
    print("--- Verifying validate_epoch ---\n")

    # Define the number of batches for the verification run
    NUM_VERIFY_BATCHES = 10

    # Instantiate the model and move it to the specified device
    verify_model = model_class(15).to(device)

    # Create a small subset of the validation data for quick verification
    val_batches = list(itertools.islice(iter(val_loader), NUM_VERIFY_BATCHES))
    # Concatenate the images and labels from the selected batches
    val_all_images = torch.cat([b[0] for b in val_batches])
    val_all_labels = torch.cat([b[1] for b in val_batches])
    # Create a TensorDataset and a DataLoader for the subset
    verify_val_subset_dataset = TensorDataset(val_all_images, val_all_labels)
    verify_val_subset_loader = DataLoader(verify_val_subset_dataset, batch_size=val_loader.batch_size)

    # Clone the initial weights of a specific layer to check for changes
    initial_weight = verify_model.conv_block1.block[0].weight.clone()

    print(f"Validating on {len(verify_val_subset_dataset)} images:\n")
    # Run a single validation epoch on the subset and get the outputs
    val_loss, val_accuracy = validate_epoch_fn(
        model=verify_model,
        val_loader=verify_val_subset_loader,
        loss_function=loss_function,
        device=device
    )

    # Get the weights of the same layer after the validation function has run
    validated_weight = verify_model.conv_block1.block[0].weight

    # Print the returned loss and accuracy
    print(f"Returned Validation Loss: {val_loss:.4f}")
    print(f"Returned Validation Accuracy: {val_accuracy:.2f}%\n")

    # Check if the returned loss and accuracy are of the correct data type (float)
    types_correct = isinstance(val_loss, float) and isinstance(val_accuracy, float)
    if types_correct:
        print("\nReturn Types Check:\tFunction returned a float for loss and accuracy.")
    else:
        print("\nReturn Types Check:\tFunction DID NOT return the correct data types.")

    # Check that the weights have not changed during validation
    weights_unchanged = torch.equal(initial_weight, validated_weight)
    if weights_unchanged:
        print("Weight Integrity Check:\tModel weights were not changed during validation.")
    else:
        print("Weight Integrity Check:\tModel weights WERE CHANGED.")