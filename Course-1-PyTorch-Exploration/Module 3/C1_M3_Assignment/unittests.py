from types import FunctionType
from typing import List

from dlai_grader.grading import print_feedback, test_case
from torch.utils.data import Dataset
from unittests_utils import DataloadersBattery, DatasetBattery, TransformationsBattery


def exercise_1(learner_class):
    def g():
        cases: List[test_case] = []

        class_name = "PlantsDataset"

        t = test_case()
        if not isinstance(learner_class, type):
            t.failed = True
            t.msg = f"{class_name} must be a class"
            t.want = f"a Python class called {class_name}."
            t.got = type(learner_class)
            return [t]
        cases.append(t)

        t = test_case()
        if not issubclass(learner_class, Dataset):
            t.failed = True
            t.msg = f"{class_name} must inherit from Dataset"
            t.want = Dataset
            t.got = learner_class.__base__
            return [t]
        cases.append(t)

        # region = dataset battery =
        try:
            dataset_battery = DatasetBattery(learner_class)

            # region == init method ==
            t = test_case()
            got, want, failed = dataset_battery.call_load_labels()
            if failed:
                t.failed = True
                t.msg = "The `load_labels` is not called within __init__ to populate `.labels`"
                t.want = """Check that the `load_labels` method is called within the `__init__` method to populate `.labels`."""
                t.got = "`.labels` is not populated correctly."
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = dataset_battery.call_read_classname()
            if failed:
                t.failed = True
                t.msg = "The `read_classname` is not called within __init__ to populate `.class_names`"
                t.want = """Check that the `read_classname` method is called within the `__init__` method to populate `.class_names`."""
                t.got = "`.class_names` is not populated correctly."
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = dataset_battery.labels_len()
            if failed:
                t.failed = True
                t.msg = "The length of labels is not correct"
                t.want = f"""The length of labels should be {want}.
                Check that `.labels` is populated correctly by the `load_labels` method."""
                t.got = f"The length of labels is {got}, which does not match."
            cases.append(t)

            t = test_case()
            got, want, failed = dataset_battery.label_particular()
            if failed:
                t.failed = True
                t.msg = "`.labels` does not have the correct values"
                t.want = """Check that `.labels` is populated correctly by the `load_labels` method."""
                t.got = "The labels retrieved are not correct. They do not match the expected values."
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = dataset_battery.class_names_len()
            if failed:
                t.failed = True
                t.msg = "The length of class_names is not correct"
                t.want = f"""The length of class_names should be {want}.
                Check that `.class_names` is populated correctly by the `read_classname` method."""
                t.got = f"The length of class_names is {got}, which does not match."
            cases.append(t)

            t = test_case()
            got, want, failed = dataset_battery.description_particular()
            if failed:
                t.failed = True
                t.msg = "`.class_names` does not have the correct values"
                t.want = """Check that `.class_names` is populated correctly by the `read_classname` method."""
                t.got = "The class names retrieved are not correct. They do not match the expected values."
                return cases + [t]
            cases.append(t)
            # endregion==

            # region== len and get_item methods ==
            t = test_case()
            got, want, failed = dataset_battery.len_method()
            if failed:
                t.failed = True
                t.msg = "The __len__ method does not return the correct value"
                t.want = f"""The __len__ method should return {want}.
                Check that the __len__ method returns the length of `.labels`."""
                t.got = f"The __len__ method returns {got}, which does not match."
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = dataset_battery.call_retrieve_image()
            if failed:
                t.failed = True
                t.msg = "The `retrieve_image` method is not called within __getitem__ to get the image"
                t.want = """Check that the `retrieve_image` method is called within the `__getitem__` method to get the image."""
                t.got = "The image is not retrieved correctly in __getitem__."
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = dataset_battery.get_item_img_type()
            if failed:
                t.failed = True
                t.msg = "The __getitem__ method does not return the correct image type"
                t.want = """The image returned by __getitem__ should be a PIL Image.
                Check that `__getitem__` uses the `retrieve_image` method to get the image."""
                t.got = f"The image returned by __getitem__ is of type {got}, which does not match."
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = dataset_battery.get_item_img_shape()
            if failed:
                t.failed = True
                t.msg = (
                    "The __getitem__ method does not return the corresponding `idx` image. "
                    "The shape of the image is not correct"
                )
                t.want = """Check that `__getitem__` uses the `.retrieve_image` method to get the image."""
                t.got = "For the provided index, the image shape returned by __getitem__ does not match the expected shape."
            cases.append(t)

            t = test_case()
            got, want, failed = dataset_battery.get_item_label()
            if failed:
                t.failed = True
                t.msg = (
                    "The __getitem__ method does not return the corresponding `idx` label. "
                    "The label value is not correct"
                )
                t.want = """Check that `__getitem__` returns the correct label from `.labels`."""
                t.got = "For the provided index, the label returned by __getitem__ does not match the expected label."
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = dataset_battery.get_item_transform()
            if failed:
                t.failed = True
                t.msg = "The __getitem__ method does not apply the transform correctly"
                t.want = """Check that `__getitem__` applies the transform if it is provided."""
                t.got = "The transform does not seem to be applied correctly in __getitem__."
                return cases + [t]
            cases.append(t)

            # endregion==

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when instantiating {class_name} or calling its methods"
            t.want = f"{class_name} to be instantiated and its methods to be called without exceptions."
            t.got = f"An exception {e} was raised."
            return cases + [t]

        # endregion=
        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases: List[test_case] = []

        func_name = "get_transformations"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        # region = dataset battery =
        try:
            transformations_battery = TransformationsBattery(learner_func)

            # region == main transform ==
            t = test_case()
            got, want, failed = transformations_battery.main_transform_len()
            if failed:
                t.failed = True
                t.msg = (
                    "The main transform has not the correct number of transformations"
                )
                t.want = f"""The main transform should have {want} transformations.
                Check that the main transform includes: Resize, ToTensor and Normalize transformations."""
                t.got = f"The main transform has {got} transformations."
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = transformations_battery.first_transform_main()
            if failed:
                t.failed = True
                t.msg = "The first transformation in the main transform is not correct"
                t.want = f"""The first transformation in the main transform should be of type {want[0]} and with size {want[1]}.
                Check that the main transform includes a Resize transformation as the first step with the correct size."""
                t.got = f"The first transformation in the main transform is of type {got[0]} and size {got[1]}."
            cases.append(t)

            t = test_case()
            got, want, failed = transformations_battery.second_transform_main()
            if failed:
                t.failed = True
                t.msg = "The second transformation in the main transform is not correct"
                t.want = f"""The second transformation in the main transform should be of type {want}.
                Check that the main transform includes a ToTensor transformation as the second step."""
                t.got = (
                    f"The second transformation in the main transform is of type {got}."
                )
            cases.append(t)

            t = test_case()
            got, want, failed = transformations_battery.third_transform_main()
            if failed:
                t.failed = True
                t.msg = "The third transformation in the main transform is not correct"
                t.want = f"""The third transformation in the main transform should be of type {want[0]} with mean {want[1]} and std {want[2]}.
                Check that the main transform includes a Normalize transformation as the third step with the correct mean and std."""
                t.got = f"The third transformation in the main transform is of type {got[0]} with mean {got[1]} and std {got[2]}."
                return cases + [t]
            cases.append(t)
            # endregion==

            # region== transform with augmentation ==
            t = test_case()
            got, want, failed = transformations_battery.augm_transform_len()
            if failed:
                t.failed = True
                t.msg = "The transform with augmentation has not the correct number of transformations"
                t.want = f"""The transform with augmentation should have {want} transformations.
                Check that the transform with augmentation includes: RandomHorizontalFlip, RandomRotation, Resize, ToTensor and Normalize transformations."""
                t.got = f"The transform with augmentation has {got} transformations."
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = transformations_battery.first_transform_augm()
            if failed:
                t.failed = True
                t.msg = "The first transformation in the transform with augmentation is not correct"
                t.want = f"""The first transformation in the transform with augmentation should be of type {want[0]} with p={want[1]}.
                Check that the transform with augmentation includes a RandomHorizontalFlip transformation as the first step with the correct probability."""
                t.got = f"The first transformation in the transform with augmentation is of type {got[0]} with p={got[1]}."
            cases.append(t)

            t = test_case()
            got, want, failed = transformations_battery.second_transform_augm()
            if failed:
                t.failed = True
                t.msg = "The second transformation in the transform with augmentation is not correct"
                t.want = f"""The second transformation in the transform with augmentation should be of type {want[0]} with degrees={want[1]}.
                Check that the transform with augmentation includes a RandomRotation transformation as the second step with the correct degrees."""
                t.got = f"The second transformation in the transform with augmentation is of type {got[0]} with degrees={got[1]}."
            cases.append(t)

            t = test_case()
            got, want, failed = transformations_battery.third_transform_augm()
            if failed:
                t.failed = True
                t.msg = "The third transformation in the transform with augmentation is not correct"
                t.want = f"""The third transformation in the transform with augmentation should be of type {want[0]} and with size {want[1]}.
                Check that the transform with augmentation includes a Resize transformation as the third step with the correct size."""
                t.got = f"The third transformation in the transform with augmentation is of type {got[0]} and size {got[1]}."
            cases.append(t)

            t = test_case()
            got, want, failed = transformations_battery.fourth_transform_augm()
            if failed:
                t.failed = True
                t.msg = "The fourth transformation in the transform with augmentation is not correct"
                t.want = f"""The fourth transformation in the transform with augmentation should be of type {want}.
                Check that the transform with augmentation includes a ToTensor transformation as the fourth step."""
                t.got = f"The fourth transformation in the transform with augmentation is of type {got}."
            cases.append(t)

            t = test_case()
            got, want, failed = transformations_battery.fifth_transform_augm()
            if failed:
                t.failed = True
                t.msg = "The fifth transformation in the transform with augmentation is not correct"
                t.want = f"""The fifth transformation in the transform with augmentation should be of type {want[0]} with mean {want[1]} and std {want[2]}.
                Check that the transform with augmentation includes a Normalize transformation as the fifth step with the correct mean and std."""
                t.got = f"The fifth transformation in the transform with augmentation is of type {got[0]} with mean {got[1]} and std {got[2]}."
                return cases + [t]
            cases.append(t)
            # endregion==

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when calling {func_name} or inspecting the returned transforms"
            t.want = f"{func_name} to be called and the returned transforms to be inspected without exceptions."
            t.got = f"An exception {e} was raised."
            return cases + [t]
        # endregion=
        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(learner_func, dataset: Dataset):
    def g():
        cases: List[test_case] = []

        func_name = "get_dataloaders"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{func_name} must be a function"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        cases.append(t)

        # region = dataloaders battery =
        try:
            dataloaders_battery = DataloadersBattery(learner_func, dataset)

            t = test_case()
            got, want, failed = dataloaders_battery.loaders_type()
            if failed:
                t.failed = True
                t.msg = "The DataLoaders are not of the correct type"
                t.want = f"The DataLoaders should be of type {want}."
                t.got = f"The DataLoaders are of type {got}."
                return cases + [t]
            cases.append(t)

            # region== dataset splits ==
            t = test_case()
            got, want, failed = dataloaders_battery.random_split_and_subset()
            if failed:
                t.failed = True
                t.msg = "The dataset splits are not Subset objects"
                t.want = (
                    "Each dataset split (train, val, test) should be a Subset object."
                    "Check that `random_split` is used to split the dataset."
                )
                t.got = "The dataset splits are not Subset objects."
            cases.append(t)

            t = test_case()
            got, want, failed = dataloaders_battery.split_sizes()
            if failed:
                t.failed = True
                t.msg = "The dataset splits do not have the correct sizes"
                t.want = f"""The dataset splits should have sizes {want}.
                Check that the dataset is split correctly according to the specified sizes by using `random_split`."""
                t.got = f"The dataset splits have sizes {got}."
                return cases + [t]
            cases.append(t)
            # endregion==

            # region== dataloader transforms ==
            t = test_case()
            got, want, failed = dataloaders_battery.train_transform()
            if failed:
                t.failed = True
                t.msg = "The training dataset does not have the correct transform"
                t.want = f"""The training dataset should use the `augmentation_transform`: {want}.  
                Check that the training dataset is wrapped with `SubsetWithTransform` using `augmentation_transform`."""
                t.got = f"The training dataset transform is {got}."
            cases.append(t)

            t = test_case()
            got, want, failed = dataloaders_battery.test_transform()
            if failed:
                t.failed = True
                t.msg = "The test dataset does not have the correct transform"
                t.want = f"""The test dataset should use the `main_transform`: {want}.  
                Check that the test dataset is wrapped with `SubsetWithTransform` using the `main_transform`."""
                t.got = f"The test dataset transform is {got}."
                return cases + [t]
            cases.append(t)
            # endregion==

            # region== dataloader shuffling ==

            t = test_case()
            got, want, failed = dataloaders_battery.train_shuffle()
            if failed:
                t.failed = True
                t.msg = "The training DataLoader does not shuffle the data"
                t.want = """The training DataLoader should shuffle the data.
                Check that the training DataLoader is created with `shuffle=True`."""
                t.got = "The training DataLoader does not shuffle the data."
            cases.append(t)

            t = test_case()
            got, want, failed = dataloaders_battery.val_shuffle()
            if failed:
                t.failed = True
                t.msg = "The validation DataLoader shuffles the data"
                t.want = """The validation DataLoader should not shuffle the data.
                Check that the validation DataLoader is created with `shuffle=False`."""
                t.got = "The validation DataLoader shuffles the data."
            cases.append(t)

            t = test_case()
            got, want, failed = dataloaders_battery.test_shuffle()
            if failed:
                t.failed = True
                t.msg = "The test DataLoader shuffles the data"
                t.want = """The test DataLoader should not shuffle the data.
                Check that the test DataLoader is created with `shuffle=False`."""
                t.got = "The test DataLoader shuffles the data."
                return cases + [t]
            cases.append(t)

            # endregion ==

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"An exception was raised when calling {func_name} or inspecting the returned DataLoaders"
            t.want = f"{func_name} to be called and the returned DataLoaders to be inspected without exceptions."
            t.got = f"An exception {e} was raised."
            return cases + [t]

        # endregion =
        return cases

    cases = g()
    print_feedback(cases)
