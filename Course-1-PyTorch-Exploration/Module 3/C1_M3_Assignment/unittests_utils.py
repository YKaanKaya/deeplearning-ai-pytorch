from unittest.mock import patch

import PIL
from torch.utils.data import Subset, dataloader, sampler
from torchvision import transforms


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


def check_method_call_init(class_name, method_name, *args, **kwargs):
    with patch.object(class_name, method_name, autospec=True) as mocked:
        obj = class_name(*args, **kwargs)
        return mocked.called


def check_method_call(
    class_name,
    outer_method,
    inner_method,
    init_args=None,
    init_kwargs=None,
    outer_args=None,
    outer_kwargs=None,
):
    """
    Checks if calling outer_method() on an instance of class_name
    results in inner_method being called.

    - init_args/init_kwargs: arguments for class initialization
    - outer_args/outer_kwargs: arguments for outer_method
    """
    init_args = init_args or ()
    init_kwargs = init_kwargs or {}
    outer_args = outer_args or ()
    outer_kwargs = outer_kwargs or {}

    with patch.object(class_name, inner_method, autospec=True) as mocked:
        obj = class_name(*init_args, **init_kwargs)
        getattr(obj, outer_method)(*outer_args, **outer_kwargs)
        return mocked.called


# region = Exercise 1 - CustomDataset =
class DatasetBattery(TestBattery):
    def __init__(self, learner_class):
        super().__init__(learner_class)

    def _get_reference_inputs(self):
        self.root_dir = "./plants_dataset"

        # to be used in some tests
        self.main_transform = transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()]
        )

    def extract_info(self):
        # create the dataset
        self.learner_dataset = self.learner_object(
            root_dir=self.root_dir, transform=None
        )

        self.ref_idx = 1000

    def get_reference_checks(self):
        self.reference_checks = {
            "labels_len": 3000,
            "label_particular": 10,
            "class_names_len": 30,
            "description_particular": "galangal",
            "len_method": 3000,
            "get_item_img_shape": (673, 379),
            "get_item_img_type": PIL.Image.Image,
            "get_item_label": 10,
            "get_item_transform": (3, 128, 128),
            "call_load_labels": True,
            "call_read_classname": True,
            "call_retrieve_image": True,
        }

    def call_load_labels(self):
        got = check_method_call_init(
            class_name=self.learner_object,
            method_name="load_labels",
            root_dir=self.root_dir,
        )
        name_check = "call_load_labels"
        return self._check(name_check, got)

    def call_read_classname(self):
        got = check_method_call_init(
            class_name=self.learner_object,
            method_name="read_classname",
            root_dir=self.root_dir,
        )
        name_check = "call_read_classname"
        return self._check(name_check, got)

    def labels_len(self):
        learner_labels = self.learner_dataset.labels
        got = len(learner_labels)

        name_check = "labels_len"
        return self._check(name_check, got)

    def label_particular(self):
        got = self.learner_dataset.labels[self.ref_idx]

        name_check = "label_particular"
        return self._check(name_check, got)

    def class_names_len(self):
        got = len(self.learner_dataset.class_names)

        name_check = "class_names_len"
        return self._check(name_check, got)

    def description_particular(self):
        label = self.learner_dataset.labels[self.ref_idx]
        got = self.learner_dataset.get_label_description(label)
        name_check = "description_particular"
        return self._check(name_check, got)

    def len_method(self):
        got = len(self.learner_dataset)
        name_check = "len_method"
        return self._check(name_check, got)

    def call_retrieve_image(self):
        got = check_method_call(
            class_name=self.learner_object,
            outer_method="__getitem__",
            inner_method="retrieve_image",
            init_args=(self.root_dir,),
            outer_args=(self.ref_idx,),
        )
        name_check = "call_retrieve_image"
        return self._check(name_check, got)

    def get_item_img_shape(self):
        img, label = self.learner_dataset[self.ref_idx]
        got = img.size  # PIL image size is (width, height)
        name_check = "get_item_img_shape"
        return self._check(name_check, got)

    def get_item_img_type(self):
        img, label = self.learner_dataset[self.ref_idx]
        got = type(img)  # PIL image size is (width, height)
        name_check = "get_item_img_type"
        return self._check(name_check, got)

    def get_item_label(self):
        img, label = self.learner_dataset[self.ref_idx]
        got = label
        name_check = "get_item_label"
        return self._check(name_check, got)

    def get_item_transform(self):
        self.learner_dataset.transform = self.main_transform
        img, label = self.learner_dataset[self.ref_idx]
        got = img.shape  # torch tensor shape is (channels, height, width)
        name_check = "get_item_transform"
        return self._check(name_check, got)


# endregion =


# region = Exercise 2 - get_transformations =
class TransformationsBattery(TestBattery):
    def __init__(self, learner_function):
        super().__init__(learner_function)

    def _get_reference_inputs(self):
        self.mean = [0.6659, 0.6203, 0.4784]
        self.std = [0.2119, 0.2155, 0.2567]

    def extract_info(self):
        self.main_transform_learner, self.augm_transform_learner = self.learner_object(
            self.mean, self.std
        )

    def get_reference_checks(self):
        self.reference_checks = {
            "main_transform_len": 3,
            "first_transform_main": (transforms.Resize, (128, 128)),
            "second_transform_main": transforms.ToTensor,
            "third_transform_main": (
                transforms.Normalize,
                self.mean,
                self.std,
            ),
            "augm_transform_len": 5,
            "first_transform_augm": (transforms.RandomVerticalFlip, 0.5),
            "second_transform_augm": (transforms.RandomRotation, [-15, 15]),
            "third_transform_augm": (transforms.Resize, (128, 128)),
            "fourth_transform_augm": transforms.ToTensor,
            "fifth_transform_augm": (
                transforms.Normalize,
                self.mean,
                self.std,
            ),
        }

    def main_transform_len(self):
        got = len(self.main_transform_learner.transforms)
        name_check = "main_transform_len"
        return self._check(name_check, got)

    def first_transform_main(self):
        first_transform = self.main_transform_learner.transforms[0]
        type_transform = type(first_transform)
        size = first_transform.size
        got = (type_transform, size)
        name_check = "first_transform_main"
        return self._check(name_check, got)

    def second_transform_main(self):
        second_transform = self.main_transform_learner.transforms[1]
        type_transform = type(second_transform)
        got = type_transform
        name_check = "second_transform_main"
        return self._check(name_check, got)

    def third_transform_main(self):
        third_transform = self.main_transform_learner.transforms[2]
        type_transform = type(third_transform)
        mean_transform = third_transform.mean
        std_transform = third_transform.std
        got = (type_transform, mean_transform, std_transform)
        name_check = "third_transform_main"
        return self._check(name_check, got)

    def augm_transform_len(self):
        got = len(self.augm_transform_learner.transforms)
        name_check = "augm_transform_len"
        return self._check(name_check, got)

    def first_transform_augm(self):
        first_transform = self.augm_transform_learner.transforms[0]
        type_transform = type(first_transform)
        p = first_transform.p
        got = (type_transform, p)
        name_check = "first_transform_augm"
        return self._check(name_check, got)

    def second_transform_augm(self):
        second_transform = self.augm_transform_learner.transforms[1]
        type_transform = type(second_transform)
        degrees = second_transform.degrees
        got = (type_transform, degrees)
        name_check = "second_transform_augm"
        return self._check(name_check, got)

    def third_transform_augm(self):
        third_transform = self.augm_transform_learner.transforms[2]
        type_transform = type(third_transform)
        size = third_transform.size
        got = (type_transform, size)
        name_check = "third_transform_augm"
        return self._check(name_check, got)

    def fourth_transform_augm(self):
        fourth_transform = self.augm_transform_learner.transforms[3]
        type_transform = type(fourth_transform)
        got = type_transform
        name_check = "fourth_transform_augm"
        return self._check(name_check, got)

    def fifth_transform_augm(self):
        fifth_transform = self.augm_transform_learner.transforms[4]
        type_transform = type(fifth_transform)
        mean_transform = fifth_transform.mean
        std_transform = fifth_transform.std
        got = (type_transform, mean_transform, std_transform)
        name_check = "fifth_transform_augm"
        return self._check(name_check, got)


# endregion =


# region = Exercise 3 - get_dataloaders =
class DataloadersBattery(TestBattery):
    def __init__(self, learner_function, learner_dataset):
        self.learner_dataset = learner_dataset
        super().__init__(learner_function)

    def _get_reference_inputs(self):
        self.main_transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.6659, 0.6203, 0.4784],
                    std=[0.2119, 0.2155, 0.2567],
                ),
            ]
        )

        self.augm_transform = transforms.Compose(
            [
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.6659, 0.6203, 0.4784],
                    std=[0.2119, 0.2155, 0.2567],
                ),
            ]
        )

        self.batch_size = 64
        self.val_fraction = 0.2
        self.test_fraction = 0.1

    def extract_info(self):
        self.lrnr_train_loader, self.lrnr_val_loader, self.lrnr_test_loader = (
            self.learner_object(
                dataset=self.learner_dataset,
                batch_size=self.batch_size,
                val_fraction=self.val_fraction,
                test_fraction=self.test_fraction,
                main_transform=self.main_transform,
                augmentation_transform=self.augm_transform,
            )
        )

    def get_reference_checks(self):
        self.reference_checks = {
            "loaders_type": (dataloader.DataLoader,) * 3,
            "split_sizes": (2100, 600, 300),
            "random_split": (
                Subset,
                Subset,
                Subset,
            ),
            "train_transform": self.augm_transform,
            "test_transform": self.main_transform,
            "train_shuffle": sampler.RandomSampler,
            "val_shuffle": sampler.SequentialSampler,
            "test_shuffle": sampler.SequentialSampler,
        }

    def loaders_type(self):
        got = (
            type(self.lrnr_train_loader),
            type(self.lrnr_val_loader),
            type(self.lrnr_test_loader),
        )
        name_check = "loaders_type"
        return self._check(name_check, got)

    def split_sizes(self):
        train_size = len(self.lrnr_train_loader.dataset)
        val_size = len(self.lrnr_val_loader.dataset)
        test_size = len(self.lrnr_test_loader.dataset)
        got = (train_size, val_size, test_size)
        name_check = "split_sizes"
        return self._check(name_check, got)

    def random_split_and_subset(self):
        train_dataset_split = self.lrnr_train_loader.dataset.subset
        val_dataset_split = self.lrnr_val_loader.dataset.subset
        test_dataset_split = self.lrnr_test_loader.dataset.subset

        got = (
            type(train_dataset_split),
            type(val_dataset_split),
            type(test_dataset_split),
        )
        name_check = "random_split"
        return self._check(name_check, got)

    def train_transform(self):
        train_dataset_transform = self.lrnr_train_loader.dataset.transform
        got = train_dataset_transform
        name_check = "train_transform"
        return self._check(name_check, got)

    def test_transform(self):
        test_dataset_transform = self.lrnr_test_loader.dataset.transform
        got = test_dataset_transform
        name_check = "test_transform"
        return self._check(name_check, got)

    def train_shuffle(self):
        got = type(self.lrnr_train_loader.sampler)
        name_check = "train_shuffle"
        return self._check(name_check, got)

    def val_shuffle(self):
        got = type(self.lrnr_val_loader.sampler)
        name_check = "val_shuffle"
        return self._check(name_check, got)

    def test_shuffle(self):
        got = type(self.lrnr_test_loader.sampler)
        name_check = "test_shuffle"
        return self._check(name_check, got)


# endregion =
