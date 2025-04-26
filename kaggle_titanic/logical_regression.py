# Create a logical regression model to predict the survival of passengers on the Titanic

import torch
import torch.nn as nn
import pandas as pd

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


# parse csv file

def parse_csv(file_path):
    dataFrame = pd.read_csv(file_path)
    # drop columns
    dataFrame = dataFrame.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    # fill missing values
    dataFrame['Age'] = dataFrame['Age'].fillna(dataFrame['Age'].mean())
    dataFrame['Embarked'] = dataFrame['Embarked'].fillna(dataFrame['Embarked'].mode()[0])
    # convert categorical variables to numerical
    dataFrame