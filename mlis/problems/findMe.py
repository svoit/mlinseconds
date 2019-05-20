# There are random function from 8 inputs and X random inputs added.
# We split data in 2 parts, on first part you will train and on second
# part we will test
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.solution = solution
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = solution.hidden_size
        self.lr = solution.lr
        self.momentum = solution.momentum

        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.linear1_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear1_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear1_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)

        self.batch_norm1 = nn.BatchNorm1d(self.hidden_size, track_running_stats=False)
        self.batch_norm1_1 = nn.BatchNorm1d(self.hidden_size, track_running_stats=False)
        self.batch_norm1_2 = nn.BatchNorm1d(self.hidden_size, track_running_stats=False)
        self.batch_norm1_3 = nn.BatchNorm1d(self.hidden_size, track_running_stats=False)
        self.batch_norm2 = nn.BatchNorm1d(self.output_size, track_running_stats=False)

        if self.solution.grid_search.enabled:
            torch.manual_seed(solution.random)

    def forward(self, x):
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)

        x = self.linear1_1(x)
        x = self.batch_norm1_1(x)
        x = F.relu(x)

        x = self.linear1_2(x)
        x = self.batch_norm1_2(x)
        x = F.relu(x)

        x = self.linear1_3(x)
        x = self.batch_norm1_3(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = self.batch_norm2(x)
        x = torch.sigmoid(x)

        return x

    def calc_loss(self, output, target):
        bce_loss = nn.BCELoss()
        loss = bce_loss(output, target)
        return loss

    def calc_predict(self, output):
        predict = output.round()
        return predict

class Solution():
    def __init__(self):
        self.lr = 0.1
        self.lr_grid = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5]

        self.hidden_size = 25
        self.hidden_size_grid = [15, 20, 25, 30, 35, 40, 45, 50, 55]

        self.momentum = 0.7
        self.random = 0
        self.error = 0.55
        self.batch_size = 128

        self.grid_search = gs.GridSearch(self).set_enabled(False)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # Put model in train mode
        model.train()
        number_of_batches = int(train_data.size(0)/self.batch_size)
        batches_counter = 0
        optimizer = optim.SGD(model.parameters(), lr=model.lr, momentum=model.momentum)
        while True:
            batch_index = step % number_of_batches
            data_batch = train_data[self.batch_size*batch_index:self.batch_size*(batch_index+1)]
            target_batch = train_target[self.batch_size*batch_index:self.batch_size*(batch_index+1)]
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1:
                break
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data_batch)
            output = model(data_batch)
            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(target_batch.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            # No more mini-batches left, stop training
            loss = model.calc_loss(output, target_batch)
            residue = (output.data-target_batch.data).abs()
            if residue.max() < self.error:
                batches_counter += 1
                if batches_counter >= number_of_batches:
                    break
            else:
                batches_counter = 0
            if model.solution.grid_search.enabled:
                self.grid_search.log_step_value('loss', loss.item(), step)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
            self.print_stats(step, loss, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
        return step
    
    def print_stats(self, step, loss, correct, total):
        if step % 100 == 0:
            print("Step = {} Prediction = {}/{} Error = {}".format(step, correct, total, loss.item()))

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 1000000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, data_size, input_size, random_input_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size
        function_input = torch.ByteTensor(function_size, input_size)
        for i in range(function_input.size(0)):
            fun_ind = i
            for j in range(function_input.size(1)):
                input_bit = fun_ind&1
                fun_ind = fun_ind >> 1
                function_input[i][j] = input_bit
        function_output = torch.ByteTensor(function_size).random_(0, 2)

        if data_size % function_size != 0:
            raise "Data gen error"

        data_input = torch.ByteTensor(data_size, input_size).view(-1, function_size, input_size)
        target = torch.ByteTensor(data_size).view(-1, function_size)
        for i in range(data_input.size(0)):
            data_input[i] = function_input
            target[i] = function_output
        data_input = data_input.view(data_size, input_size)
        target = target.view(data_size)
        if random_input_size > 0:
            data_random = torch.ByteTensor(data_size, random_input_size).random_(0, 2)
            data = torch.cat([data_input, data_random], dim=1)
        else:
            data = data_input
        perm = torch.randperm(data.size(1))
        data = data[:,perm]
        perm = torch.randperm(data.size(0))
        data = data[perm]
        target = target[perm]
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        data_size = 256*32
        input_size = 8
        random_input_size = min(32, (case-1)*4)

        data, target = self.create_data(2*data_size, input_size, random_input_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs and {} random inputs".format(input_size, random_input_size))

class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=-1)
