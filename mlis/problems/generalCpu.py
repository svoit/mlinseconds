# You need to learn a function with n inputs.
# For given number of inputs, we will generate random function.
# Your task is to learn it
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
        self.hidden_size = solution.hidden_size
        self.lr = solution.lr

        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.linear1_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear1_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear1_4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, output_size)

        self.batch_norm1 = nn.BatchNorm1d(self.hidden_size, track_running_stats=False)
        self.batch_norm1_2 = nn.BatchNorm1d(self.hidden_size, track_running_stats=False)
        self.batch_norm1_3 = nn.BatchNorm1d(self.hidden_size, track_running_stats=False)
        self.batch_norm1_4 = nn.BatchNorm1d(self.hidden_size, track_running_stats=False)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu6(x)
        x = self.batch_norm1(x)

        x = self.linear1_2(x)
        x = F.relu6(x)
        x = self.batch_norm1_2(x)

        x = self.linear1_3(x)
        x = F.relu6(x)
        x = self.batch_norm1_3(x)

        x = self.linear1_4(x)
        x = F.relu6(x)
        x = self.batch_norm1_4(x)

        x = self.linear2(x)
        x = F.sigmoid(x)

        return x

    def calc_loss(self, output, target):
        loss = F.binary_cross_entropy(output, target)
        return loss

    def calc_predict(self, output):
        predict = output.round()
        return predict


class Solution():
    def __init__(self):
        self.lr = 0.009
        self.lr_grid = [0.0001, 0.001, 0.005, 0.01, 0.1, 1]

        self.hidden_size = 50
        self.hidden_size_grid = [15, 20, 25, 30, 35, 40, 45, 50, 55]

        self.grid_search = gs.GridSearch(self).set_enabled(False)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # Put model in train mode
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=model.lr)
        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1 or (model.solution.grid_search.enabled and step > 10):
                break
            data = train_data
            target = train_target
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(data)
            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            if total == correct:
                break
            # calculate loss
            loss = model.calc_loss(output, target)
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
        if step % 1000 == 0:
            print("Step = {} Prediction = {}/{} Error = {}".format(step, correct, total, loss.item()))

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 10000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, input_size, seed):
        random.seed(seed)
        data_size = 1 << input_size
        data = torch.FloatTensor(data_size, input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):
            for j in range(input_size):
                input_bit = (i>>j)&1
                data[i,j] = float(input_bit)
            target[i] = float(random.randint(0, 1))
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        input_size = min(3+case, 7)
        data, target = self.create_data(input_size, case)
        return sm.CaseData(case, Limits(), (data, target), (data, target)).set_description("{} inputs".format(input_size))


class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=-1)
