from types import FunctionType

from dlai_grader.grading import print_feedback, test_case
from unittests_utils import (
    DataLoaderBattery,
    EvaluateBattery,
    ModelBattery,
    TrainBattery,
)


def exercise_1(learner_func):
    def g():
        cases = []

        t_type = test_case()
        failed = not isinstance(learner_func, FunctionType)
        if failed:
            t_type.failed = True
            t_type.msg = "create_dataloaders has incorrect type"
            t_type.want = FunctionType
            t_type.got = type(learner_func)
            return [t_type]
        cases.append(t_type)

        try:
            dataloader_battery = DataLoaderBattery(learner_func)

            t = test_case()
            got, want, failed = dataloader_battery.train_loader_batch_size()
            if failed:
                t.failed = True
                t.msg = "Incorrect batch_size of train_loader. Please make sure you are setting the batch_size as batch_size=batch_size, and not hardcoding it"
                t.want = "batch_size=batch_size"
                t.got = f"batch_size={got}"
            cases.append(t)

            t = test_case()
            got, want, failed = dataloader_battery.test_loader_batch_size()
            if failed:
                t.failed = True
                t.msg = "Incorrect batch_size of test_loader. Please make sure you are setting the batch_size as batch_size=batch_size, and not hardcoding it"
                t.want = "batch_size=batch_size"
                t.got = f"batch_size={got}"
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = dataloader_battery.train_loader_length()
            if failed:
                t.failed = True
                t.msg = "Incorrect length of train_loader. Please make sure you are using train_dataset when setting up train_loader"
                t.want = want
                t.got = got
            cases.append(t)

            t = test_case()
            got, want, failed = dataloader_battery.test_loader_length()
            if failed:
                t.failed = True
                t.msg = "Incorrect length of test_loader. Please make sure you are using test_dataset when setting up test_loader"
                t.want = want
                t.got = got
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = dataloader_battery.train_loader_shuffle()
            if failed:
                t.failed = True
                t.msg = "Incorrect shuffle of train_loader. Please make sure you are setting the shuffle as True for the train_loader"
                t.want = "shuffle=True"
                t.got = "shuffle=False or shuffle=None"
                cases.append(t)

            t = test_case()
            got, want, failed = dataloader_battery.test_loader_shuffle()
            if failed:
                t.failed = True
                t.msg = "Incorrect shuffle of test_loader. Please make sure you are setting the shuffle as False for the test_loader"
                t.want = "shuffle=False"
                t.got = "shuffle=True or shuffle=None"
                return cases + [t]
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Error when calling create_emnist_dataloaders: {e}"
            t.want = "No exception"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_2(learner_func):
    def g():
        cases = []

        t_type = test_case()
        failed = not isinstance(learner_func, FunctionType)
        if failed:
            t_type.failed = True
            t_type.msg = "emnist_init_model has incorrect type"
            t_type.want = FunctionType
            t_type.got = type(learner_func)
            return [t_type]
        cases.append(t_type)

        try:
            model_battery = ModelBattery(learner_func)

            t = test_case()
            got, want, failed = model_battery.model_type()
            if failed:
                t.failed = True
                t.msg = "The model has incorrect type. Please make sure you are returning a torch.nn.Sequential model"
                t.want = want
                t.got = got
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = model_battery.model_num_layers()
            if failed:
                t.failed = True
                t.msg = "model has an incorrect number of layers"
                t.want = "between 1 and 7 layers (inclusive), including the fixed first and final layers"
                t.got = got
            cases.append(t)

            t = test_case()
            got, want, failed = model_battery.first_layer_type()
            if failed:
                t.failed = True
                t.msg = "First layer is not an nn.Flatten layer"
                t.want = want
                t.got = got
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = model_battery.last_layer_type()
            if failed:
                t.failed = True
                t.msg = "Last layer is not an nn.Linear layer"
                t.want = want
                t.got = got
                return cases + [t]
            cases.append(t)

            checks_mlt = model_battery.middle_layers_type()
            for layer_num, (got, want, failed) in enumerate(checks_mlt, 1):
                t = test_case()
                if failed:
                    t.failed = True
                    t.msg = f"Layer {layer_num + 1} ({layer_num}) is not of type nn.Linear or nn.ReLU"
                    t.want = "nn.Linear or nn.ReLU"
                    t.got = got
                    return cases + [t]
                cases.append(t)

            t = test_case()
            got, want, failed = model_battery.loss_function_type()
            if failed:
                t.failed = True
                t.msg = "The loss function has incorrect type"
                t.want = want
                t.got = got
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = model_battery.optimizer_type()
            if failed:
                t.failed = True
                t.msg = "The optimizer has incorrect type"
                t.want = want
                t.got = got
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = model_battery.learning_rate()
            if failed:
                t.failed = True
                t.msg = "The optimizer has an incorrect learning rate"
                t.want = want
                t.got = got
                return cases + [t]
            cases.append(t)

            checks_hli = model_battery.hidden_layers_inputs()
            for layer_num, (got, want, failed) in enumerate(checks_hli, 1):
                t = test_case()
                if failed:
                    t.failed = True
                    t.msg = f"Layer shape incompatibility in layer {layer_num} ({layer_num-1})"
                    t.want = want
                    t.got = got
                    return cases + [t]
                cases.append(t)

            checks_hlo = model_battery.hidden_layers_outputs()
            for layer_num, (got, want, failed) in enumerate(checks_hlo, 1):
                t = test_case()
                if failed:
                    t.failed = True
                    t.msg = (
                        f"Layer {layer_num} ({layer_num-1}) exceeds 256 hidden units"
                    )
                    t.want = "Hidden layer size less than or equal to 256 units"
                    t.got = got
                    return cases + [t]
                cases.append(t)

            t = test_case()
            got, want, failed = model_battery.num_classes()
            if failed:
                t.failed = True
                t.msg = "The final layer has incorrect output size. Please make sure you are passing in the num_classes and not hardcoding it"
                t.want = want
                t.got = got
                return cases + [t]

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Error when calling emnist_init_model: {e}"
            t.want = "No exception"
            t.got = f"Exception: {e}"

        return cases

    cases = g()
    print_feedback(cases)


def exercise_3(
    learner_func,
    learner_model,
    learner_loss_func,
    learner_optimizer,
    learner_train_loader,
):
    def g():
        cases = []

        t_type = test_case()
        failed = not isinstance(learner_func, FunctionType)
        if failed:
            t_type.failed = True
            t_type.msg = "train_epoch has incorrect type"
            t_type.want = FunctionType
            t_type.got = type(learner_func)
            return cases + [t_type]
        cases.append(t_type)

        try:
            train_battery = TrainBattery(
                learner_func,
                model=learner_model,
                loss_function=learner_loss_func,
                optimizer=learner_optimizer,
                train_loader=learner_train_loader,
            )

            t = test_case()
            checks_methods = train_battery.required_methods_check()

            for check in checks_methods:
                t = test_case()
                got, want, failed = check
                method = want
                if failed:
                    t.failed = True
                    t.msg = f"{method} is not used in train_epoch"
                    t.want = f"{method} usage in train_epoch as instructed"
                    t.got = f"{method} not found"
                    return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = train_battery.model_type()
            if failed:
                t.failed = True
                t.msg = "train_epoch did not return a Sequential model"
                t.want = want
                t.got = got
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = train_battery.train_check()
            loss_0, loss_1 = got
            if failed:
                t.failed = True
                t.msg = "Average loss did not change after running train_epoch two times. Make sure you are calculating the loss correctly."
                t.want = "Different average loss values"
                t.got = f"Loss 1: {loss_0}, Loss 2: {loss_1}"
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = train_battery.batch_size()
            if failed:
                t.failed = True
                t.msg = "Model output has incorrect batch size."
                t.want = want
                t.got = got
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = train_battery.output_shape()
            if failed:
                t.failed = True
                t.msg = "Model output has incorrect shape."
                t.want = want
                t.got = got
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = train_battery.batch_size_1_sample()
            if failed:
                t.failed = True
                t.msg = "Model output has incorrect batch size."
                t.want = want
                t.got = got
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = train_battery.output_shape_1_sample()
            if failed:
                t.failed = True
                t.msg = "Model output has incorrect shape."
                t.want = want
                t.got = got
                return cases + [t]

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Error when calling train_emnist_model: {e}"
            t.want = "No exception"
            t.got = f"Exception: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_4(learner_func, learner_model, learner_test_loader):
    def g():
        cases = []

        t_type = test_case()
        failed = not isinstance(learner_func, FunctionType)
        if failed:
            t_type.failed = True
            t_type.msg = "evaluate has incorrect type"
            t_type.want = FunctionType
            t_type.got = type(learner_func)
            return [t_type]
        cases.append(t_type)

        try:
            evaluate_battery = EvaluateBattery(
                learner_func, model=learner_model, data_loader=learner_test_loader
            )

            t = test_case()
            got, want, failed = evaluate_battery.no_grad_present()
            if failed:
                t.failed = True
                t.msg = "torch.no_grad() is not used in evaluate"
                t.want = "torch.no_grad() usage in evaluate as instructed"
                t.got = "torch.no_grad() not found"
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = evaluate_battery.output_shape()
            if failed:
                t.failed = True
                t.msg = "Model output has incorrect shape."
                t.want = want
                t.got = got
                return cases + [t]
            cases.append(t)

            t = test_case()
            got, want, failed = evaluate_battery.output_shape_1_sample()
            if failed:
                t.failed = True
                t.msg = "Model output for single sample has incorrect shape."
                t.want = want
                t.got = got
                return cases + [t]

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"evaluate raised an exception"
            t.want = "The accuracy percentage of the model on the test dataset."
            t.got = f'Evaluation ran into an error: "{e}"'
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


def exercise_5(class_accuracies):
    """
    Checks if each alphabet has an accuracy of 0.60 or greater.

    Args:
        class_accuracies (dict): A dictionary where keys are letters (alphabets) and
                                 values are their corresponding accuracies (floats).
    """

    def g():
        cases = []

        failed_letters = []
        for letter, accuracy in class_accuracies.items():
            if accuracy < 0.60:
                failed_letters.append(f"{letter} as {(accuracy * 100):.2f}% accurate")

        t = test_case()
        if failed_letters:
            t.failed = True
            t.msg = 'One or more letters have accuracy below 60%.\n\n- Try training the model again (run the "Train Your Model" code cell again). If that doesn’t improve the accuracy,\n- Try training for a larger number of epochs\n- Try a different model architecture (in Exercise 2) to train with\n'
            t.want = "All letters with accuracy >= 60%"
            t.got = ", ".join(failed_letters)
            return cases + [t]
        cases.append(t)

        t = test_case()
        t.msg = "All letters have accuracy >= 60%"
        t.want = "All letters with accuracy >= 60%"
        t.got = "All letters with accuracy >= 60%"
        return [t]

    cases = g()
    print_feedback(cases)
