import os
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from datasets.simple_dataset import SimpleDataSet


def save_model(filename):
    torch.save(model.state_dict(), store_model_path+filename+".pt")


datetime_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
dataset = "sample"
model_type = "big_model"
train_filename = "scaled_train_out.csv"

batch_size = 32
learning_rate = 1e-5
n_epochs = 2000
save_epochs = 100

model_name = "{model_type}_batch_{batch_size}_lr_{learning_rate}_{datetime_str}".format(
    model_type=model_type,
    batch_size=batch_size,
    learning_rate=learning_rate,
    datetime_str=datetime_str
)

data_path = '../../data/'+dataset+"/"
runs_path = '../../runs/'+dataset+"_"+model_name
store_model_path = data_path+"models/"+model_name+"/"

print("reading data...")
data = SimpleDataSet(data_path+train_filename)
batch_size = len(data) if batch_size == "ALL" else batch_size

for path in [runs_path, store_model_path]:
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


data_size = len(data)
train_size = round(data_size*.8)
test_size = data_size - train_size

train_set, test_set = torch.utils.data.random_split(
    data, [train_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=True)

model = Net(data.input_count, data.label_count)
model.to(device)


criterion = nn.CrossEntropyLoss()
criterion.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_loss = 0
train_loss_min = np.Inf
test_loss = 0

writer = SummaryWriter(log_dir=runs_path)

for epoch in range(n_epochs+1):
    train_acc = 0
    for i, data in enumerate(train_loader, 0):
        x_var, y_var = data
        x_var, y_var = x_var.to(device), y_var.to(device)

        optimizer.zero_grad()
        output = model(x_var.float())

        values, labels = torch.max(output, 1)

        loss = criterion(output, torch.max(y_var.long(), 1)[1])
        loss.backward()
        optimizer.step()

        train_labels = torch.max(y_var, 1)[1]

        train_num_correct = torch.eq(labels.data, train_labels)
        train_acc += torch.sum(labels == train_labels)
        train_loss += loss.item() * batch_size

    train_loss = train_loss / train_size
    train_total_accuracy = train_acc / train_size

    writer.add_scalar("Accuracy/train", train_total_accuracy, epoch)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.flush()

    if train_loss < train_loss_min:
        train_loss_min = train_loss

    if epoch % save_epochs == 0:
        pass
        save_model("model_{epoch}".format(epoch=epoch))

    # test
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            x_var, y_var = data
            x_var, y_var = x_var.to(device), y_var.to(device)

            output = model(x_var.float())

            # loss = criterion(output, y_var.long())
            loss = criterion(output, torch.max(y_var.long(), 1)[1])
            values, labels = torch.max(output, 1)

            test_labels = torch.max(y_var, 1)[1]

            test_num_correct = torch.eq(labels.data, test_labels)
            test_loss = loss.item()
            test_acc = torch.sum(labels == test_labels) / len(y_var)

            writer.add_scalar("Accuracy/test", test_acc, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.flush()

    model.train()


save_model("model_final")
print('Training Ended! ')
print(device)

