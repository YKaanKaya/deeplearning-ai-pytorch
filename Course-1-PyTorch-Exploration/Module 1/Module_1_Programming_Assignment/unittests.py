import inspect
import math
import re
from types import FunctionType

import torch
import torch.nn as nn
import torch.optim as optim

from dlai_grader.grading import test_case, print_feedback
import unittests_utils



def exercise_1(learner_func):
    def g():
        cases = []
        
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "rush_hour_feature has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        
        
        sample_tensor = torch.tensor([
            [5.74,      16.59,       0,          29.06],
            [8.80,      12.25,       0,          23.94],
            [15.36,     11.76,       1,          32.40],
            [2.46,      14.44,       0,          14.09]
        ], dtype=torch.float32)
        
        sample_hours = sample_tensor[:, 1]
        sample_weekends = sample_tensor[:, 2]
        
        learner_sample = learner_func(sample_hours, sample_weekends)
        
        # ##### Return type check #####
        t = test_case()
        if not isinstance(learner_sample, torch.Tensor):
            t.failed = True
            t.msg = "Incorrect is_rush_hour_mask type returned from rush_hour_feature"
            t.want = torch.Tensor
            t.got = type(learner_sample)
            return [t]
        
        # ##### Shape check #####
        t = test_case()
        if learner_sample.shape != (4, ):
            t.failed = True
            t.msg = "is_rush_hour_mask returned from rush_hour_feature has wrong shape. Follow the exercise instructions to make sure you are correctly implementing all of the conditions and operations"
            t.want = "torch.Size([4])"
            t.got = learner_sample.shape
            return [t]
        
        sample_tensor = torch.tensor([
            [5.74,      16.59,       1,          29.06],
            [10.66,     16.07,       0,          37.17],
            [5.35,      9.42,        1,          17.06],
        ], dtype=torch.float32)
        
        sample_hours = sample_tensor[:, 1]
        sample_weekends = sample_tensor[:, 2]
        
        learner_sample = learner_func(sample_hours, sample_weekends)
        
        
        expected = torch.tensor([0., 1., 0.])
        
        # ##### Expected Values Check #####
        t = test_case()
        if not torch.equal(learner_sample, expected):
            t.failed = True
            t.msg = "rush_hour_feature returned incorrect values. Follow the exercise instructions to make sure you are correctly implementing all of the conditions and operations"
            t.want = expected
            t.got = learner_sample
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)



def exercise_2(learner_func):
    def g():
        cases = []
        
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "prepare_data has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        
        
        file_path = './data_with_features.csv'
        subset_df = unittests_utils.load_rows(file_path)
        
        learner_features, learner_targets, learner_results = learner_func(subset_df)
        
        # ##### Return type check (prepared_features) #####
        t = test_case()
        if not isinstance(learner_features, torch.Tensor):
            t.failed = True
            t.msg = "Incorrect prepared_features type returned from prepare_data"
            t.want = torch.Tensor
            t.got = type(learner_features)
            return [t]
        
        # ##### Return type check (prepared_targets) #####
        t = test_case()
        if not isinstance(learner_targets, torch.Tensor):
            t.failed = True
            t.msg = "Incorrect prepared_targets type returned from prepare_data"
            t.want = torch.Tensor
            t.got = type(learner_targets)
            return [t]
        
        # ##### DType check (full_tensor) #####
        t = test_case()
        if learner_results["full_tensor"].dtype != torch.float32:
            t.failed = True
            t.msg = "Incorrect dtype for full_tensor"
            t.want = torch.float32
            t.got = learner_results["full_tensor"].dtype
            return [t]
        
        expected_raw_distances = torch.tensor([16.3400, 18.0300,  7.0400,  3.0900,  5.3300,  9.1200, 16.5400, 17.3500, 1.1300, 10.7000])
        expected_raw_hours = torch.tensor([ 8.3200,  9.4900,  8.0000,  9.7000, 19.5900,  8.1700,  8.0000,  8.0000, 17.1000,  8.0000])       
        expected_raw_weekends = torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
        expected_raw_targets = torch.tensor([59.6100, 70.4600, 31.5500,  9.0600,  8.2900, 21.1800, 61.1300, 64.8400, 7.7300, 45.4200])
        
        expected_values = [expected_raw_distances, expected_raw_hours, expected_raw_weekends, expected_raw_targets]
        
        # ##### Returned column (slicing) values checks #####
        keys_to_check = ['raw_distances', 'raw_hours', 'raw_weekends', 'raw_targets']
        for key, expected_tensor in zip(keys_to_check, expected_values):
            learner_tensor = learner_results[key]
            t = test_case()
            if not torch.equal(learner_tensor, expected_tensor):
                t.failed = True
                t.msg = f"The tensor for '{key}' is incorrect. Make sure you are correctly slicing to separate out {key} column"
                t.want = expected_tensor
                t.got = learner_tensor
                cases.append(t)
                
        # ##### Return cases, if any, before moving on #####    
        if cases:
            return cases
        
        source_code = inspect.getsource(learner_func)
        source_code = unittests_utils.remove_comments(source_code)
        
        # ##### Check if "rush_hour_feature" function is being used #####
        required_function = "rush_hour_feature"
        pattern = r'\b' + required_function + r'\b'
        
        t = test_case()
        if not re.search(pattern, source_code):
            t.failed = True
            t.msg = f"{required_function} is not used in prepare_data"
            t.want = f"{required_function} usage in prepare_data as instructed."
            t.got = f"{required_function} not found."
            return [t]
        
        # ##### Checking if "unsqueeze(1)" has been applied to *_col variables #####
        keys_to_check = ['distances_col', 'hours_col', 'weekends_col', 'rush_hour_col']
        for key in keys_to_check:            
            t = test_case()
            if learner_results[key].shape != torch.Size([10, 1]):
                t.failed = True
                t.msg = f"Incorrect shape for '{key}'. Make sure you are applying unsqueeze(1) to it"
                t.want = torch.Size([10, 1])
                t.got = learner_results[key].shape
                cases.append(t)
            
        # ##### Return cases, if any, before moving on #####    
        if cases:
            return cases
        
        
        expected_distances_col = expected_raw_distances.unsqueeze(1)
        expected_hours_col = expected_raw_hours.unsqueeze(1)
        expected_weekends_col = expected_raw_weekends.unsqueeze(1)
        # 2D column vector
        expected_rush_hour_col = torch.tensor([[1.], 
                                               [1.], 
                                               [1.], 
                                               [1.], 
                                               [0.], 
                                               [0.], 
                                               [1.], 
                                               [1.], 
                                               [1.], 
                                               [1.]])
        
        expected_col_values = [expected_distances_col, expected_hours_col, expected_weekends_col, expected_rush_hour_col]
        
        # ##### Checking if *_col variables have expected values #####    
        keys_to_check = ['distances_col', 'hours_col', 'weekends_col', 'rush_hour_col']
        for key, expected_tensor in zip(keys_to_check, expected_col_values):
            learner_tensor = learner_results[key]
            t = test_case()
            if not torch.equal(learner_tensor, expected_tensor):
                t.failed = True
                t.msg = f"The tensor values for '{key}' are incorrect. Make sure you using the correct raw values tensor needed to reshape the 1D feature tensor into 2D column vector {key}"
                t.want = expected_tensor
                t.got = learner_tensor
            cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
    
    
    
def exercise_3(learner_func):
    def g():
        cases = []
        
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "init_model has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        model, optimizer, loss_function = learner_func()
        
        
        # ##### Return Type Check (Model) #####
        t = test_case()
        if not isinstance(model, nn.Sequential):
            t.failed = True
            t.msg = "Incorrect model type returned from init_model"
            t.want = nn.Sequential
            t.got = type(model)
            return [t]
        
        # ##### Return Type Check (optimizer) #####
        t = test_case()
        if not isinstance(optimizer, optim.SGD):
            t.failed = True
            t.msg = "Incorrect optimizer type returned from init_model"
            t.want = optim.SGD
            t.got = type(optimizer)
            return [t]
        
        # ##### Return Type Check (loss_function) #####
        t = test_case()
        if not isinstance(loss_function, nn.MSELoss):
            t.failed = True
            t.msg = "Incorrect loss_function type returned from init_model"
            t.want = nn.MSELoss
            t.got = type(loss_function)
            return [t]
        
        # ##### Total Number of model's layers Check #####
        t = test_case()
        if len(model) != 5:
            t.failed = True
            t.msg = "model has an incorrect number of layers"
            t.want = 5
            t.got = len(model)
            return [t]
        
        
        # ##### Check if model's layers are as expected #####
        layers_list = [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
        
        for layer_num, layer in enumerate(model):
            t = test_case()
            if not isinstance(layer, layers_list[layer_num]):
                t.failed = True
                t.msg = f"model's ({layer_num}) layer is incorrect"
                t.want = layers_list[layer_num]
                t.got = layer
                cases.append(t)
            
        # ##### Return cases, if any, before moving on #####    
        if cases:
            return cases
        
        
        # ##### Check model's Linear layers are of expected dimension #####
        layer_dims = [[4, 64], 0, [64, 32], 0, [32, 1]]

        for layer_num, layer in enumerate(model):
            # Check if the remainder when dividing by 2 is 0
            if layer_num % 2 == 0:
                t = test_case()
                if layer.in_features != layer_dims[layer_num][0] or layer.out_features != layer_dims[layer_num][1]:
                    t.failed = True
                    t.msg = f"({layer_num}): Linear layer has incorrect dimensions"
                    t.want = f"Linear({layer_dims[layer_num][0]}, {layer_dims[layer_num][1]})"
                    t.got = f"Linear({layer.in_features}, {layer.out_features})"
                cases.append(t)
        
        
        # ##### Learning Rate Value Check #####
        lr = optimizer.defaults["lr"]
        t = test_case()
        if lr != 0.01:
            t.failed = True
            t.msg = "incorrect learning rate set in optimizer"
            t.want = 0.01
            t.got = lr
        cases.append(t)

        
        return cases

    cases = g()
    print_feedback(cases)
    


def exercise_4(learner_func, features, targets):
    def g():
        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "train_model has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        source_code = inspect.getsource(learner_func)
        source_code = unittests_utils.remove_comments(source_code)
        
        # ##### Check if these methods are being used in the implementation #####
        required_methods = ["optimizer.zero_grad()", "loss.backward()", "optimizer.step()"]
        for method in required_methods:
            t = test_case()
            if method not in source_code:
                t.failed = True
                t.msg = f"{method} is not used in train_model"
                t.want = f"{method} usage in train_model as instructed"
                t.got = f"{method} not found"
                return [t]


        try:
            learner_trained_model, learner_loss = learner_func(features, targets, 15000, verbose=False)
            
            # ##### Check return type of model #####
            t = test_case()
            if not isinstance(learner_trained_model, nn.Sequential):
                t.failed = True
                t.msg = "train_model did not return a Sequential model"
                t.want = nn.Sequential
                t.got = type(learner_trained_model)
                return [t]
            
            # ##### Check if loss is changing #####
            t = test_case()
            if learner_loss[0] == learner_loss[2]:
                t.failed = True
                t.msg = "Loss did not change during taining. Make sure you are calculating the loss correctly"
                t.want = "Different loss values"
                t.got = f"Loss at 5000th epoch: {learner_loss[0]}, Loss at 15000th epoch: {learner_loss[2]}"
                return [t]
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"train_model raised an exception"
            t.want = f"The model to train."
            t.got = f"Training ran into an error: \"{e}\""
            return [t]
        

        # ##### Define an input for the model #####
        inputs = torch.tensor([[-0.0824, -0.3469,  1.0000,  0.0000]])
        with torch.no_grad():
            outputs = learner_trained_model(inputs)
            
        
        # ##### Check model output shape #####
        t = test_case()
        if outputs.shape[0] != 1:
            t.failed = True
            t.msg = "model output has incorrect shape"
            t.want = 1
            t.got = outputs.shape[0]
            return [t]
        
        
        expected_output = 21.649423599243164
        
        # ##### Check expected value
        prediction = outputs[0].item()
        t = test_case()
        close = math.isclose(prediction, expected_output, abs_tol=0.4)
        if not close:
            t.failed = True
            t.msg = f"model's output is not close enough to expected output. Make sure your loss is decreasing as expected"
            t.want = f"{expected_output} +- 0.4"
            t.got = prediction
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)