import copy

from IPython.display import display
from ipywidgets import Output
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.models as tv_models
from torchvision import transforms
from tqdm.notebook import tqdm



def course_2_preview(train_dataset, val_dataset, loss_function, device, num_epochs):
    """
    Trains and validates a machine learning model on a given dataset.

    Args:
        train_dataset: The dataset used for training.
        val_dataset: The dataset used for validation.
        loss_function: The loss function to use for training.
        device: The device (e.g., 'cuda' or 'cpu') to run the model on.
        num_epochs: The total number of epochs to train for.

    Returns:
        - model: The best performing model instance.
    """
    # Define the mean and standard deviation for ImageNet normalization.
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Create a composition of transforms for the training data, including augmentation.
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    # Create a composition of transforms for the validation data.
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    # Apply the defined transformations to the datasets.
    train_dataset.transform = train_transform
    val_dataset.transform = val_transform

    # Set the batch size for the DataLoaders.
    batch_size = 64
    
    # Create a DataLoader for the training dataset.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create a DataLoader for the validation dataset.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the MobileNetV3 Small model without pre-trained weights.
    model = tv_models.mobilenet_v3_small(weights=None)
    
    # Load a pre-trained state dictionary from a local file.
    state_dict = torch.load("./c2_preview/mobilenet_v3_small-047dcff4.pth", map_location=torch.device('cpu'))
    
    # Load the state dictionary into the model.
    model.load_state_dict(state_dict)

    # Determine the number of classes from the training dataset.
    num_classes = len(train_dataset.classes)
    
    # Get the number of input features for the final classifier layer.
    in_features = model.classifier[3].in_features
    
    # Replace the final classifier layer to match the number of classes in the dataset.
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    # Move the model to the specified compute device.
    model.to(device)

    # Initialize the AdamW optimizer with model parameters and a learning rate.
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Initialize a learning rate scheduler that adjusts the learning rate based on a cosine annealing schedule.
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Initialize variables to track the best model's performance.
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    
    # Start the main training loop.
    for epoch in range(num_epochs):
        # Set the model to training mode.
        model.train()
        running_loss = 0.0
        
        # Iterate over batches of training data with a progress bar.
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for images, labels in train_pbar:
            # Move images and labels to the specified compute device.
            images, labels = images.to(device), labels.to(device)
            
            # Reset the gradients of the optimizer.
            optimizer.zero_grad()
            
            # Perform a forward pass.
            outputs = model(images)
            
            # Calculate the loss.
            loss = loss_function(outputs, labels)
            
            # Perform backpropagation to compute gradients.
            loss.backward()
            
            # Update the model's weights.
            optimizer.step()
            
            # Accumulate the running loss.
            running_loss += loss.item() * images.size(0)
            
        # Calculate the average loss for the epoch.
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Set the model to evaluation mode.
        model.eval()
        running_val_loss = 0.0
        correct, total = 0, 0
        
        # Disable gradient calculations for validation to save memory and computations.
        with torch.no_grad():
            # Iterate over batches of validation data with a progress bar.
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for images, labels in val_pbar:
                # Move images and labels to the specified compute device.
                images, labels = images.to(device), labels.to(device)
                
                # Perform a forward pass.
                outputs = model(images)
                
                # Calculate the validation loss.
                val_loss = loss_function(outputs, labels)
                
                # Accumulate the running validation loss.
                running_val_loss += val_loss.item() * images.size(0)
                
                # Get the predicted class indices.
                _, predicted = torch.max(outputs, 1)
                
                # Update the total number of labels.
                total += labels.size(0)
                
                # Update the number of correctly predicted labels.
                correct += (predicted == labels).sum().item()
                
        # Calculate the average validation loss for the epoch.
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        # Calculate the validation accuracy for the epoch.
        epoch_accuracy = 100.0 * correct / total
        
        # Print the training and validation metrics for the epoch.
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_accuracy:.2f}%")
        
        # Update the learning rate scheduler.
        scheduler.step()
        
        # Check if the current model has the best validation accuracy so far.
        if epoch_accuracy > best_val_accuracy:
            # Update the best validation accuracy.
            best_val_accuracy = epoch_accuracy
            
            # Save the epoch number.
            best_epoch = epoch + 1
            
            # Save a deep copy of the model's state dictionary.
            best_model_state = copy.deepcopy(model.state_dict())

    # After the training loop, check if a best model state was saved.
    if best_model_state:
        # Print a message indicating the performance of the best model.
        print(f"\n--- Returning best model with {best_val_accuracy:.2f}% validation accuracy, achieved at epoch {best_epoch} ---")
        
        # Load the state of the best model.
        model.load_state_dict(best_model_state)
    
    # Return the best model
    return model