import inspect
import math
import os
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


def get_train_test():
    # Define the path where the EMNIST data will be stored
    data_path = "./EMNIST_data"

    # Check if the data folder exists to avoid re-downloading
    if os.path.exists(data_path) and os.path.isdir(data_path):
        download = False
    else:
        download = True

    # Precomputed mean and std for EMNIST Letters dataset
    mean = (0.1736,)
    std = (0.3317,)

    # Create a transform that converts images to tensors and normalizes them
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts images to PyTorch tensors and scales pixel values to [0, 1]
            transforms.Normalize(
                mean=mean, std=std
            ),  # Applies normalization using the computed mean and std
        ]
    )

    # Load the EMNIST Letters training set
    train_dataset = datasets.EMNIST(
        root=data_path,
        split="letters",
        train=True,
        download=False,
        transform=transform,
    )

    test_dataset = datasets.EMNIST(
        root=data_path,
        split="letters",
        train=False,
        download=False,
        transform=transform,
    )
    return train_dataset, test_dataset


# region = General Check structure =


class TestBattery:
    def __init__(self, learner_object):
        self.learner_object = learner_object

        self._get_reference_inputs()
        self.extract_info()
        self.get_reference_checks()

    def _get_reference_inputs(self):
        pass

    def extract_info(self):
        pass

    def get_reference_checks(self):
        pass

    def _create_reference_checks(self):
        checks_dict = {}
        if self.reference_checks is not None:
            for key in self.reference_checks.keys():
                check_fcn = getattr(self, f"{key}", None)
                # run the corresponding method
                got, want, failed = check_fcn()
                checks_dict[key] = got

        return checks_dict

    def _check(self, check_name, got):
        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition


# endregion


# region = Exercise 1 =
def check_shuffle(data_loader, should_shuffle):
    """
    Checks if a DataLoader is shuffling data by observing the order of labels.
    """
    first_iteration_labels = []
    for _, labels in data_loader:
        first_iteration_labels.extend(labels.tolist())

    second_iteration_labels = []
    for _, labels in data_loader:
        second_iteration_labels.extend(labels.tolist())

    if should_shuffle:
        return first_iteration_labels != second_iteration_labels
    else:
        return first_iteration_labels == second_iteration_labels


class DataLoaderBattery(TestBattery):

    def _get_reference_inputs(self):
        self.train_dataset, self.test_dataset = get_train_test()
        self.batch_size = 32

    def extract_info(self):
        self.train_loader, self.test_loader = self.learner_object(
            self.train_dataset, self.test_dataset, batch_size=self.batch_size
        )

    def get_reference_checks(self):
        self.reference_checks = {
            # "train_loader_type": None,
            # "test_loader_type": None,
            "train_loader_batch_size": 32,
            "test_loader_batch_size": 32,
            "train_loader_length": 3900,
            "test_loader_length": 650,
            "train_loader_shuffle": True,
            "test_loader_shuffle": False,
        }

    def train_loader_batch_size(self):

        got = self.train_loader.batch_size

        name_check = "train_loader_batch_size"
        return self._check(name_check, got)

    def test_loader_batch_size(self):
        got = self.test_loader.batch_size

        name_check = "test_loader_batch_size"
        return self._check(name_check, got)

    def train_loader_length(self):
        got = len(self.train_loader)

        name_check = "train_loader_length"
        return self._check(name_check, got)

    def test_loader_length(self):
        got = len(self.test_loader)

        name_check = "test_loader_length"
        return self._check(name_check, got)

    def train_loader_shuffle(self):
        got = check_shuffle(self.train_loader, should_shuffle=True)

        name_check = "train_loader_shuffle"
        return self._check(name_check, got)

    def test_loader_shuffle(self):
        got = check_shuffle(self.test_loader, should_shuffle=True)

        name_check = "test_loader_shuffle"
        return self._check(name_check, got)


# endregion =


# region = Exercise 2 =
class ModelBattery(TestBattery):
    def _get_reference_inputs(self):
        self.num_classes = 26
        self.input_size = 784  # 28*28

    def extract_info(self):
        self.model, self.loss_function, self.optimizer = self.learner_object(
            num_classes=self.num_classes
        )

    def get_reference_checks(self):
        self.reference_checks = {
            "model_num_layers": True,
            "model_type": nn.Sequential,
            "first_layer_type": nn.Flatten,
            "last_layer_type": nn.Linear,
            "loss_function_type": nn.CrossEntropyLoss,
            "optimizer_type": optim.Adam,
            "learning_rate": 0.001,
            "num_classes": self.num_classes,
        }

    def model_type(self):
        got = type(self.model)

        name_check = "model_type"
        return self._check(name_check, got)

    def model_num_layers(self):
        got = 1 <= len(self.model) <= 7

        name_check = "model_num_layers"
        return self._check(name_check, got)

    def first_layer_type(self):
        first_layer = self.model[0]
        got = type(first_layer)

        name_check = "first_layer_type"
        return self._check(name_check, got)

    def last_layer_type(self):
        last_layer = self.model[-1]
        got = type(last_layer)

        name_check = "last_layer_type"
        return self._check(name_check, got)

    def middle_layers_type(self):
        middle_layers = self.model[1:-1]
        checks = []

        for layer in middle_layers:
            got = type(layer)
            want = [nn.Linear, nn.ReLU]
            failed = got not in want
            checks.append((got, want, failed))

        return checks

    def loss_function_type(self):
        got = type(self.loss_function)

        name_check = "loss_function_type"
        return self._check(name_check, got)

    def optimizer_type(self):
        got = type(self.optimizer)

        name_check = "optimizer_type"
        return self._check(name_check, got)

    def learning_rate(self):
        got = self.optimizer.defaults["lr"]

        name_check = "learning_rate"
        return self._check(name_check, got)

    def hidden_layers_inputs(self):
        checks = []
        input_size = self.input_size

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                got = layer.in_features
                want = input_size
                failed = got != want
                checks.append((got, want, failed))
            input_size = layer.out_features
        return checks

    def hidden_layers_outputs(self):
        last_layer = self.model[-1]
        checks = []
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                got = layer.out_features
                want = True
                failed = (got <= 0) or (layer == last_layer)
                checks.append((got, want, failed))
        return checks

    def num_classes(self):
        last_layer = self.model[-1]
        got = last_layer.out_features

        name_check = "num_classes"
        return self._check(name_check, got)


# endregion =


# region = Exercise 3 =
def remove_comments(code):
    # This regex pattern matches comments in the code
    pattern = r"#.*"

    # Use re.sub() to replace comments with an empty string
    code_without_comments = re.sub(pattern, "", code)

    # Split the code into lines, strip each line, and filter out empty lines
    lines = code_without_comments.splitlines()
    non_empty_lines = [line.rstrip() for line in lines if line.strip()]

    # Join the non-empty lines back into a single string
    return "\n".join(non_empty_lines)


class TrainBattery(TestBattery):
    def __init__(self, learner_object, model, loss_function, optimizer, train_loader):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_loader = train_loader

        super().__init__(learner_object)

    def _get_reference_inputs(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.inputs = torch.randn(2, 1, 28, 28).to(self.device)

        self.inputs_1_sample = torch.randn(1, 1, 28, 28).to(self.device)

    def extract_info(self):
        self.trained_model, self.loss_0 = self.learner_object(
            self.model,
            self.loss_function,
            self.optimizer,
            self.train_loader,
            device=self.device,
            verbose=False,
        )

        with torch.no_grad():
            self.outputs = self.model(self.inputs)
            self.outputs_1_sample = self.model(self.inputs_1_sample)

    def get_reference_checks(self):
        self.reference_checks = {
            "model_type": nn.Sequential,
            "batch_size": 2,
        }

    # no reference for this one
    def required_methods_check(self):
        checks = []

        source_code = inspect.getsource(self.learner_object)
        source_code = remove_comments(source_code)

        required_methods = [
            "optimizer.zero_grad()",
            "loss.backward()",
            "optimizer.step()",
        ]

        for method in required_methods:
            got = method in source_code
            want = method
            failed = not got
            checks.append((got, want, failed))
        return checks

    def model_type(self):
        got = type(self.trained_model)
        name_check = "model_type"
        return self._check(name_check, got)

    # no reference for this one
    def train_check(self):
        # Train for a second time and check that the loss decreases
        second_train, loss_1 = self.learner_object(
            self.trained_model,
            self.loss_function,
            self.optimizer,
            self.train_loader,
            device=self.device,
            verbose=False,
        )

        got = (self.loss_0, loss_1)

        rel_tol = 1e-5

        want = None

        failed = math.isclose(loss_1, self.loss_0, rel_tol=rel_tol)

        return got, want, failed

    def batch_size(self):
        got = self.outputs.shape[0]
        name_check = "batch_size"
        return self._check(name_check, got)

    def output_shape(self):
        got = self.outputs.shape
        want = torch.Size([2, 26])
        failed = len(got) != len(want) or (got[1] != want[1])
        return got, want, failed

    def batch_size_1_sample(self):
        got = self.outputs_1_sample.shape[0]
        want = 1
        failed = got != want
        return got, want, failed

    def output_shape_1_sample(self):
        got = self.outputs_1_sample.shape
        want = torch.Size([1, 26])
        failed = len(got) != len(want) or (got[1] != want[1])
        return got, want, failed


# endregion =

# region = Exercise 4 =


class EvaluateBattery(TestBattery):
    def __init__(self, learner_object, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        super().__init__(learner_object)

    def _get_reference_inputs(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.inputs = torch.randn(2, 1, 28, 28).to(self.device)

        self.inputs_1_sample = torch.randn(1, 1, 28, 28).to(self.device)

    def extract_info(self):
        self.model = self.model.to(self.device)

        # Apply the evaluation function to get accuracy
        self.accuracy = self.learner_object(
            self.model, self.data_loader, device=self.device, verbose=False
        )

        with torch.no_grad():
            self.outputs = self.model(self.inputs)
            self.outputs_1_sample = self.model(self.inputs_1_sample)

    def get_reference_checks(self):
        self.reference_checks = {
            "no_grad": None,
            "output_shape": torch.Size([2, 26]),
            "output_shape_1_sample": torch.Size([1, 26]),
        }

    def no_grad_present(self):
        source_code = inspect.getsource(self.learner_object)
        source_code = remove_comments(source_code)

        got = "with torch.no_grad()" in source_code
        want = None
        failed = not got
        return got, want, failed

    def output_shape(self):
        got = self.outputs.shape
        name_check = "output_shape"
        return self._check(name_check, got)

    def output_shape_1_sample(self):
        got = self.outputs_1_sample.shape
        name_check = "output_shape_1_sample"
        return self._check(name_check, got)


# endregion =
