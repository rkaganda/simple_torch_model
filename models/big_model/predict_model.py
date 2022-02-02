import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pandas as pd

from datasets.simple_dataset import SimpleDataSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.fc7(x)
        return x


model_file = "model_3800"
model_directory = "sample-05_20220128_0729"
dataset = "sample"
input_file = "scaled_predict_out.csv"

model_path = "../../data/{dataset}/models/{model_directory}/{model_file}.pt".format(
    dataset=dataset,
    model_directory=model_directory,
    model_file=model_file
)

store_prediction_path = "../../data/{dataset}/models/{model_directory}/{model_file}/".format(
    dataset=dataset,
    model_directory=model_directory,
    model_file=model_file
)

input_path = "../../data/{dataset}/".format(
    dataset=dataset
)

for path in [store_prediction_path]:
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)


print("reading data...")
data = SimpleDataSet(input_path+input_file)


model = Net(data.input_count, data.label_count)
model.load_state_dict(torch.load(model_path))
model.to(device)



model.eval()

predict_loader = DataLoader(data, batch_size=len(data), shuffle=False)


with torch.no_grad():
    for i, data in enumerate(predict_loader, 0):
        x_var, y_var = data
        x_var, y_var = x_var.to(device), y_var.to(device)

        output = model(x_var.float())

        sm_out = torch.softmax(output, dim=1)

        values, labels = torch.max(output, 1)

        predict_labels = torch.max(y_var, 1)[1]

        test_num_correct = torch.eq(labels.data, predict_labels)
        # print(test_num_correct)

        pred = F.softmax(output, dim=1)

        stacked = np.concatenate([np.expand_dims(labels.cpu().numpy(), axis=1), pred.cpu().numpy()], axis=1)

        df_columns = ['prediction']
        for n in range(0, stacked.shape[1]-1):
            df_columns.append('pred_'+str(n))

        df = pd.DataFrame(stacked, columns=df_columns)
        print(store_prediction_path + input_file)
        df.to_csv(store_prediction_path + input_file)
        print(df.head())


print(device)

