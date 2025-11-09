import copy
from types import FunctionType
from typing import List, Type, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from dlai_grader.grading import test_case, print_feedback



def exercise_1(learner_func):
    def g():
        cases = []
        
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "define_transformations has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        mean = (0.4, 0.4, 0.4)
        std = (0.3, 0.3, 0.3)
        expected_function_return = transforms.Compose
        
        learner_train_transform, learner_val_transform = learner_func(mean, std)

        ### Return type check 1 (train_transformations)
        t = test_case()
        if not isinstance(learner_train_transform, expected_function_return):
            t.failed = True
            t.msg = "Incorrect train_transformations return type"
            t.want = "<class 'torchvision.transforms.transforms.Compose'>"
            t.got = f"{type(learner_train_transform)}"
            return [t]
        
        ### Return type check 2 (val_transformations)
        t = test_case()
        if not isinstance(learner_val_transform, expected_function_return):
            t.failed = True
            t.msg = "Incorrect val_transformations return type"
            t.want = "<class 'torchvision.transforms.transforms.Compose'>"
            t.got = f"{type(learner_val_transform)}"
            return [t]

        # Verify the number of train transformations, should be 5
        t = test_case()
        if len(learner_train_transform.transforms) != 5:
            t.failed = True
            t.msg = f"Expected 5 train transformations, but found {len(learner_train_transform.transforms)}"
            t.want = "5 train transformations in define_transformations. RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ToTensor and Normalize"
            t.got = f"{len(learner_train_transform.transforms)} train transformations"
            return [t]
        
        # Verify the number of val transformations, should be 2
        t = test_case()
        if len(learner_val_transform.transforms) != 2:
            t.failed = True
            t.msg = f"Expected 2 val transformations, but found {len(learner_val_transform.transforms)}"
            t.want = "2 val transformations in define_transformations. ToTensor and Normalize"
            t.got = f"{len(learner_val_transform.transforms)} val transformations"
            return [t]
        
        ###################################################################################
        
        ### Check for RandomHorizontalFlip in train_transformations
        hflip_found = False

        for transform in learner_train_transform.transforms:
            if isinstance(transform, transforms.RandomHorizontalFlip):
                hflip_found = True
                break

        t = test_case()
        if hflip_found == False:
            t.failed = True
            t.msg = "RandomHorizontalFlip transform not found in train_transformations"
            t.want = "RandomHorizontalFlip transform present in train_transformations"
            t.got = "No RandomHorizontalFlip transform in train_transformations"
        cases.append(t)
        
        
        ### Check for RandomVerticalFlip in train_transformations
        vflip_found = False

        for transform in learner_train_transform.transforms:
            if isinstance(transform, transforms.RandomVerticalFlip):
                vflip_found = True
                break

        t = test_case()
        if vflip_found == False:
            t.failed = True
            t.msg = "RandomVerticalFlip transform not found in train_transformations"
            t.want = "RandomVerticalFlip transform present in train_transformations"
            t.got = "No RandomVerticalFlip transform in train_transformations"
        cases.append(t)
        
        
        ### Check for RandomRotation in train_transformations
        expected_rotation = [-15.0, 15.0]
        rotation_found = False
        found_correct_rotation = False
        found_rotation = None
        
        for transform in learner_train_transform.transforms:
            if isinstance(transform, transforms.RandomRotation):
                rotation_found = True
                found_rotation = transform.degrees
                if found_rotation == expected_rotation:
                    found_correct_rotation = True
                break

        t = test_case()
        if not rotation_found:
            t.failed = True
            t.msg = "RandomRotation transform not found in train_transformations"
            t.want = "train_transformations to include RandomRotation transform"
            t.got = "train_transformations without RandomRotation transform"
        elif not found_correct_rotation:
            t.failed = True
            t.msg = "RandomRotation found in train_transformations, but with incorrect degree"
            t.want = f"{expected_rotation}"
            t.got = f"{found_rotation}"
        cases.append(t)
        
        
        ### Check for ToTensor in train_transformations
        totensor_found = False

        for transform in learner_train_transform.transforms:
            if isinstance(transform, transforms.ToTensor):
                totensor_found = True
                break

        t = test_case()
        if totensor_found == False:
            t.failed = True
            t.msg = "ToTensor transform not found in train_transformations"
            t.want = "ToTensor transform present in train_transformations"
            t.got = "No ToTensor transform in train_transformations"
        cases.append(t)
        
        
        ### Check for Normalize in train_transformations with specific mean and std
        normalize_found = False
        found_correct_normalize = False
        found_mean = None
        found_std = None
        for transform in learner_train_transform.transforms:
            if isinstance(transform, transforms.Normalize):
                normalize_found = True
                found_mean = transform.mean
                found_std = transform.std
                if (found_mean == mean) and (found_std == std):
                    found_correct_normalize = True
                break

        t = test_case()
        if not normalize_found:
            t.failed = True
            t.msg = "Normalize transform not found in train_transformations"
            t.want = "train_transformations to include Normalize transform"
            t.got = "train_transformations without Normalize transform"
        elif not found_correct_normalize:
            t.failed = True
            t.msg = "Normalize found in train_transformations, but with incorrect mean and/or std"
            t.want = "(mean=mean, std=std)"
            t.got = f"(mean={tuple(found_mean) if found_mean is not None else None}, std={tuple(found_std) if found_std is not None else None})"
        cases.append(t)
            
        ###################################################################################
        
        ### Check for ToTensor in val_transformations
        totensor_found = False

        for transform in learner_val_transform.transforms:
            if isinstance(transform, transforms.ToTensor):
                totensor_found = True
                break

        t = test_case()
        if totensor_found == False:
            t.failed = True
            t.msg = "ToTensor transform not found in val_transformations"
            t.want = "ToTensor transform present in val_transformations"
            t.got = "No ToTensor transform in val_transformations"
        cases.append(t)
        
        ### Check for Normalize in val_transformations with specific mean and std
        normalize_found = False
        found_correct_normalize = False
        found_mean = None
        found_std = None
        for transform in learner_val_transform.transforms:
            if isinstance(transform, transforms.Normalize):
                normalize_found = True
                found_mean = transform.mean
                found_std = transform.std
                if (found_mean == mean) and (found_std == std):
                    found_correct_normalize = True
                break

        t = test_case()
        if not normalize_found:
            t.failed = True
            t.msg = "Normalize transform not found in val_transformations"
            t.want = "val_transformations to include Normalize transform"
            t.got = "val_transformations without Normalize transform"
        elif not found_correct_normalize:
            t.failed = True
            t.msg = "Normalize found in val_transformations, but with incorrect mean and/or std"
            t.want = "(mean=mean, std=std)"
            t.got = f"(mean={tuple(found_mean) if found_mean is not None else None}, std={tuple(found_std) if found_std is not None else None})"
        cases.append(t)
        
        ###################################################################################
        
        # Check the order of transformations (train_transformations)
        expected_train_order = [
            transforms.RandomHorizontalFlip,
            transforms.RandomVerticalFlip,
            transforms.RandomRotation,
            transforms.ToTensor,
            transforms.Normalize,
        ]
        learner_train_order = [type(transform) for transform in learner_train_transform.transforms]

        t = test_case()
        if learner_train_order != expected_train_order:
            t.failed = True
            t.msg = "Train transformations are not applied in the expected order"
            t.want = f"[{', '.join([t.__name__ for t in expected_train_order])}]"
            t.got = f"[{', '.join([t.__name__ for t in learner_train_order])}]"
        cases.append(t)
        
        # Check the order of transformations (val_transformations)
        expected_val_order = [
            transforms.ToTensor,
            transforms.Normalize,
        ]
        learner_val_order = [type(transform) for transform in learner_val_transform.transforms]

        t = test_case()
        if learner_val_order != expected_val_order:
            t.failed = True
            t.msg = "Validation transformations are not applied in the expected order"
            t.want = f"[{', '.join([t.__name__ for t in expected_val_order])}]"
            t.got = f"[{', '.join([t.__name__ for t in learner_val_order])}]"
        cases.append(t)
        
        return cases

    cases = g()
    print_feedback(cases)
    
    
    
def exercise_2(learner_class):
    def g():
        cases = []
        class_name = "CNNBlock"

        # Test Case 1: Check if the provided solution is a class
        t = test_case()
        if not isinstance(learner_class, type):
            t.failed = True
            t.msg = f"{class_name} has an incorrect type"
            t.want = f"A Python class named {class_name}."
            t.got = type(learner_class)
            return [t]

        # Test Case 2: Check for inheritance from nn.Module
        t = test_case()
        if learner_class.__base__ != nn.Module:
            t.failed = True
            t.msg = f"{class_name} did not inherit from the correct base class"
            t.want = nn.Module
            t.got = learner_class.__base__
            return [t]
        
        try:
            dummy_in_channels = 3
            dummy_out_channels = 64
            learner_model = learner_class(in_channels=dummy_in_channels, out_channels=dummy_out_channels)

            # Test Case 3: Check for the presence of the 'block' attribute
            t = test_case()
            if not hasattr(learner_model, 'block'):
                t.failed = True
                t.msg = f"The '{class_name}' class is missing the 'block' attribute"
                t.want = "An attribute named 'block' containing the sequential model."
                t.got = "No 'block' attribute found."
                return [t]

            # Test Case 4: Check if 'block' is an instance of nn.Sequential
            t = test_case()
            if not isinstance(learner_model.block, nn.Sequential):
                t.failed = True
                t.msg = "The 'block' attribute should be an instance of nn.Sequential"
                t.want = nn.Sequential
                t.got = type(learner_model.block)
                return [t]
            
            block_layers = list(learner_model.block.children())
            
            # Test Case 5: Check for the correct number of layers in the sequential block
            expected_num_layers = 4
            t = test_case()
            if len(block_layers) != expected_num_layers:
                t.failed = True
                t.msg = "The 'block' does not contain the correct number of layers"
                t.want = f"{expected_num_layers} (Conv2d, BatchNorm2d, ReLU and MaxPool2d) layers inside nn.Sequential."
                t.got = f"{len(block_layers)} layers found."
                return [t]
            
            # Test Case 6: Check layer types and their order
            expected_layer_types = [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d]
            actual_layer_types = [type(layer) for layer in block_layers]
            t = test_case()
            if actual_layer_types != expected_layer_types:
                t.failed = True
                t.msg = "The layers in the 'block' are not of the correct type or in the correct order"
                t.want = " -> ".join([t.__name__ for t in expected_layer_types])
                t.got = " -> ".join([t.__name__ for t in actual_layer_types])
                return [t]

            # Test Case 7: Check parameters for each layer
            # Conv2d layer parameters
            conv_layer = block_layers[0]
            t = test_case()
            if not (conv_layer.in_channels == dummy_in_channels and
                    conv_layer.out_channels == dummy_out_channels and
                    conv_layer.kernel_size == (3, 3) and
                    conv_layer.padding == (1, 1)):
                t.failed = True
                t.msg = "Parameters for nn.Conv2d are incorrect. Check in_channels, out_channels, kernel_size, and padding"
                t.want = "in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding"
                t.got = f"in_channels={conv_layer.in_channels}, out_channels={conv_layer.out_channels}, kernel_size={conv_layer.kernel_size}, padding={conv_layer.padding}"
            cases.append(t)
            
            # BatchNorm2d layer parameters
            bn_layer = block_layers[1]
            t = test_case()
            if bn_layer.num_features != dummy_out_channels:
                t.failed = True
                t.msg = "Parameter for nn.BatchNorm2d is incorrect. Its 'num_features' should match the 'out_channels' of the preceding Conv2d"
                t.want = f"num_features=out_channels"
                t.got = f"num_features={bn_layer.num_features}"
            cases.append(t)

            # MaxPool2d layer parameters
            pool_layer = block_layers[3]
            t = test_case()
            if not (pool_layer.kernel_size == 2 and pool_layer.stride == 2):
                t.failed = True
                t.msg = "Parameters for nn.MaxPool2d are incorrect. Check kernel_size and stride"
                t.want = "kernel_size=2, stride=2"
                t.got = f"kernel_size={pool_layer.kernel_size}, stride={pool_layer.stride}"
            cases.append(t)
            
            # Test Case 8: Forward pass and output shape check
            dummy_input = torch.randn(1, dummy_in_channels, 32, 32)
            learner_model.eval()
            with torch.no_grad():
                output = learner_model(dummy_input)
            
            expected_shape = torch.Size([1, dummy_out_channels, 16, 16])
            
            t = test_case()
            if not isinstance(output, torch.Tensor) or output.shape != expected_shape:
                t.failed = True
                t.msg = "The output from the forward pass is incorrect. Check your forward method and layer configurations."
                t.want = f"A tensor with shape {expected_shape}"
                t.got = f"An object of type {type(output).__name__} with shape {output.shape if isinstance(output, torch.Tensor) else 'N/A'}"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An error occurred during model testing: {e}"
            t.want = "The model to initialize and run a forward pass without errors."
            t.got = "An exception was raised."
            cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
    
    
    
def exercise_3(learner_class, cnn_block):
    def g():
        cases = []
        class_name = "SimpleCNN"

        # Test Case 1: Check if the solution is a class
        t = test_case()
        if not isinstance(learner_class, type):
            t.failed = True
            t.msg = f"{class_name} has an incorrect type."
            t.want = f"A Python class named {class_name}."
            t.got = type(learner_class)
            return [t]

        # Test Case 2: Check for correct inheritance
        t = test_case()
        if learner_class.__base__ != nn.Module:
            t.failed = True
            t.msg = f"{class_name} did not inherit from the correct base class."
            t.want = nn.Module
            t.got = learner_class.__base__
            return [t]

        try:
            dummy_num_classes = 10
            learner_model = learner_class(num_classes=dummy_num_classes)

            # Test Case 3: Check for all required attributes
            required_attrs = ['conv_block1', 'conv_block2', 'conv_block3', 'classifier']
            for attr in required_attrs:
                t = test_case()
                if not hasattr(learner_model, attr):
                    t.failed = True
                    t.msg = f"The '{class_name}' class is missing the '{attr}' attribute"
                    t.want = f"An attribute named '{attr}'."
                    t.got = "Attribute not found."
                    return [t]
            
            # Test Case 4: Check the types of the main attributes
            CNNBlock = cnn_block
            
            t = test_case()
            if not all(isinstance(getattr(learner_model, b), CNNBlock) for b in required_attrs[:3]):
                t.failed = True
                t.msg = "Convolutional blocks must be instances of the CNNBlock class"
                t.want = "conv_block1, conv_block2, conv_block3 to be of type CNNBlock."
                t.got = f"Types found: {type(learner_model.conv_block1).__name__}, " \
                        f"{type(learner_model.conv_block2).__name__}, " \
                        f"{type(learner_model.conv_block3).__name__}"
                return [t]

            t = test_case()
            if not isinstance(learner_model.classifier, nn.Sequential):
                t.failed = True
                t.msg = "The 'classifier' attribute must be an instance of nn.Sequential."
                t.want = nn.Sequential
                t.got = type(learner_model.classifier)
                return [t]
            
            # Test Case 5: Check parameters of the CNNBlocks
            block_params = [
                (learner_model.conv_block1, 3, 32, 'conv_block1'),
                (learner_model.conv_block2, 32, 64, 'conv_block2'),
                (learner_model.conv_block3, 64, 128, 'conv_block3')
            ]
            for block, in_c, out_c, name in block_params:
                 if isinstance(block, CNNBlock):
                    conv_layer = block.block[0]
                    t = test_case()
                    if conv_layer.in_channels != in_c or conv_layer.out_channels != out_c:
                        t.failed = True
                        t.msg = f"Incorrect channels for {name}. Check the in_channels and out_channels arguments"
                        t.want = f"in_channels={in_c}, out_channels={out_c}"
                        t.got = f"in_channels={conv_layer.in_channels}, out_channels={conv_layer.out_channels}"
                    cases.append(t)

            # Test Case 6: Check classifier architecture and parameters
            if isinstance(learner_model.classifier, nn.Sequential):
                classifier_layers = list(learner_model.classifier.children())
                expected_types = [nn.Flatten, nn.Linear, nn.ReLU, nn.Dropout, nn.Linear]
                actual_types = [type(layer) for layer in classifier_layers]

                t = test_case()
                if actual_types != expected_types:
                    t.failed = True
                    t.msg = "The layers in the 'classifier' are not of the correct type or in the correct order"
                    t.want = " -> ".join([t.__name__ for t in expected_types])
                    t.got = " -> ".join([t.__name__ for t in actual_types])
                    cases.append(t)
                else:
                    # Check parameters if architecture is correct
                    linear1, dropout, linear2 = classifier_layers[1], classifier_layers[3], classifier_layers[4]
                    expected_in_features = 128 * 4 * 4
                    if linear1.in_features != expected_in_features or linear1.out_features != 512:
                        t.failed = True
                        t.msg = "Parameters for the first nn.Linear layer in the classifier are incorrect"
                        t.want = f"in_features={expected_in_features}, out_features=512"
                        t.got = f"in_features={linear1.in_features}, out_features={linear1.out_features}"
                    elif dropout.p != 0.6:
                        t.failed = True
                        t.msg = "The dropout probability in the classifier is incorrect"
                        t.want = 0.6
                        t.got = dropout.p
                    elif linear2.in_features != 512 or linear2.out_features != dummy_num_classes:
                        t.failed = True
                        t.msg = "Parameters for the final nn.Linear layer are incorrect."
                        t.want = "in_features=512, out_features=num_classes"
                        t.got = f"in_features={linear2.in_features}, out_features={linear2.out_features}"
                    cases.append(t)
            
            # Test Case 7: Forward Pass Execution Order and Output Shape
            call_order = []
            def get_hook(name):
                def hook(model, input, output): call_order.append(name)
                return hook

            for name, module in learner_model.named_children():
                module.register_forward_hook(get_hook(name))
            
            # A 32x32 input image becomes 4x4 after three 2x2 pooling layers
            dummy_input = torch.randn(1, 3, 32, 32)
            learner_model.eval()
            with torch.no_grad():
                output = learner_model(dummy_input)

            expected_order = ['conv_block1', 'conv_block2', 'conv_block3', 'classifier']
            t = test_case()
            if call_order != expected_order:
                t.failed = True
                t.msg = "The forward pass does not call the modules in the correct order"
                t.want = " -> ".join(expected_order)
                t.got = " -> ".join(call_order)
            cases.append(t)
            
            expected_shape = torch.Size([1, dummy_num_classes])
            t = test_case()
            if not isinstance(output, torch.Tensor) or output.shape != expected_shape:
                t.failed = True
                t.msg = "The final output shape from the forward pass is incorrect"
                t.want = f"A tensor with shape {expected_shape}"
                t.got = f"An object of type {type(output).__name__} with shape {output.shape if isinstance(output, torch.Tensor) else 'N/A'}"
            cases.append(t)

        except Exception as e:
            # Catch any other errors during model testing
            t = test_case()
            t.failed = True
            t.msg = f"An error occurred during model instantiation or forward pass: {e}"
            t.want = "The model to initialize and run without errors."
            t.got = "An exception was raised."
            cases.append(t)
        
        return cases

    # Execute the tests and print the results
    cases = g()
    print_feedback(cases)
    
    
    
def exercise_4(learner_func):
    def g():
        cases = []
        
        # Test Case 1: Check if the solution is a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "train_epoch has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            num_epochs = 4
            
            # 1. A simple model
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = nn.Linear(10, 8)
                    self.relu = nn.ReLU()
                    self.linear2 = nn.Linear(8, 2)
                def forward(self, x):
                    return self.linear2(self.relu(self.linear1(x)))
            
            model = SimpleModel().to(device)

            dummy_inputs = torch.randn(32, 10)
            dummy_labels = torch.randint(0, 2, (32,))
            dataset = TensorDataset(dummy_inputs, dummy_labels)
            train_loader = DataLoader(dataset, batch_size=4)
            
            loss_function = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            # --- Test the function's behavior over multiple epochs ---

            # Capture the model's weights BEFORE training
            initial_weights = copy.deepcopy(model.state_dict())
            
            # Store the loss from each epoch
            epoch_losses = []
            for _ in range(num_epochs):
                avg_loss = learner_func(model, train_loader, loss_function, optimizer, device)
                epoch_losses.append(avg_loss)

            # Capture the model's weights AFTER training
            final_weights = model.state_dict()
            
            # Test Case 2: Check the return type and value from the first epoch
            t = test_case()
            first_loss = epoch_losses[0]
            if not isinstance(first_loss, float) or first_loss < 0:
                t.failed = True
                t.msg = "The function should return a single float representing the average loss, which cannot be negative"
                t.want = "A non-negative float value."
                t.got = f"A value of type {type(first_loss).__name__}: {first_loss}"
                return [t]
            
            # Test Case 3: Check if weights changed between the start and end
            weights_changed = any(not torch.equal(initial_weights[key], final_weights[key]) for key in initial_weights)
            
            t = test_case()
            if not weights_changed:
                t.failed = True
                t.msg = ("The model's weights did not change after multiple training epochs. "
                         "Please ensure you are correctly implementing the training steps: "
                         "1. Zero gradients, 2. Compute loss, 3. Call backward(), and 4. Call optimizer.step()")
                t.want = "Model weights to be updated."
                t.got = "Model weights remained the same."
            cases.append(t)

            # Test Case 4: Check if the loss is decreasing
            t = test_case()
            initial_loss = epoch_losses[0]
            final_loss = epoch_losses[-1]
            # Check if final loss is less than initial loss
            if final_loss >= initial_loss:
                t.failed = True
                t.msg = "The training loss did not decrease over several epochs. This indicates a problem in the training loop logic"
                t.want = f"Final loss ({final_loss:.4f}) to be less than the Initial loss ({initial_loss:.4f})."
                t.got = f"The loss did not decrease. Full loss history: {[f'{l:.4f}' for l in epoch_losses]}"
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An error occurred while running your function: {e}"
            t.want = "The function to execute without errors."
            t.got = "An exception was raised."
            cases.append(t)
            
        return cases
        
    cases = g()
    print_feedback(cases)
    
    
    
def exercise_5(learner_func):
    def g():
        cases = []
        
        # Test Case 1: Check if the solution is a function
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "validate_epoch has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 1. A predictable model that will always favor class 1
            class PredictableModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    # A dummy layer to ensure the model has parameters
                    self.dummy_layer = nn.Linear(10, 2)
                def forward(self, x):
                    # Return a fixed output that always favors class 1.
                    batch_size = x.shape[0]
                    # Create a tensor of shape [batch_size, 2] where the
                    # logit for class 1 is always higher than for class 0.
                    output = torch.tensor([[0.0, 1.0]], device=x.device).repeat(batch_size, 1)
                    return output
            
            model = PredictableModel().to(device)

            # 2. Dummy data where all labels are class 1
            dummy_inputs = torch.randn(16, 10)
            dummy_labels = torch.ones(16, dtype=torch.long) # All labels are '1'
            dataset = TensorDataset(dummy_inputs, dummy_labels)
            val_loader = DataLoader(dataset, batch_size=4)
            
            # 3. Loss function
            loss_function = nn.CrossEntropyLoss()

            # --- Test the function's behavior ---

            # Capture the model's weights BEFORE validation
            initial_weights = copy.deepcopy(model.state_dict())
            
            # Execute the learner's function
            result = learner_func(model, val_loader, loss_function, device)

            # Capture the model's weights AFTER validation
            final_weights = model.state_dict()
            
            # Test Case 2: Check the return type and value ranges
            t = test_case()
            if not (isinstance(result, tuple) and len(result) == 2 and 
                    isinstance(result[0], float) and isinstance(result[1], float)):
                t.failed = True
                t.msg = "The function must return a tuple of two floats: (epoch_val_loss, epoch_accuracy)"
                t.want = "A tuple like (float, float)."
                t.got = f"A value of type {type(result).__name__}: {result}"
                return [t]
            elif not (0 <= result[1] <= 100):
                t.failed = True
                t.msg = "The returned accuracy must be a percentage between 0 and 100"
                t.want = "An accuracy value between 0 and 100."
                t.got = f"Accuracy = {result[1]}"
                return [t]
            cases.append(t)
            
            # Test Case 3: The most important check - did the weights stay the same?
            weights_are_same = all(torch.equal(initial_weights[key], final_weights[key]) for key in initial_weights)
            
            t = test_case()
            if not weights_are_same:
                t.failed = True
                t.msg = ("The model's weights changed during validation, which should not happen. "
                         "Ensure all validation logic is wrapped in a `with torch.no_grad():` block")
                t.want = "Model weights to remain unchanged."
                t.got = "Model weights were modified."
                return [t]
            cases.append(t)

            # Test Case 4: Check for correct accuracy calculation
            t = test_case()
            # With the predictable model and data, accuracy should be exactly 100.0
            expected_accuracy = 100.0
            actual_accuracy = result[1] if isinstance(result, tuple) and len(result) > 1 else -1.0
            
            if abs(actual_accuracy - expected_accuracy) > 1e-6:
                t.failed = True
                t.msg = ("The accuracy calculation appears to be incorrect. "
                         "Please check your logic for getting predictions from model outputs (e.g., using `torch.max`)")
                t.want = f"An accuracy of {expected_accuracy}% for the validation case."
                t.got = f"An accuracy of {actual_accuracy}%."
            cases.append(t)
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An error occurred while running your function: {e}"
            t.want = "The function to execute without errors."
            t.got = "An exception was raised."
            cases.append(t)
            
        return cases
        
    cases = g()
    print_feedback(cases)