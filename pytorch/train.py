import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.DataLoader import SpeckDataset
from models.model import NeuralNetwork

print("Creating Data...")
print("---" * 30)

# Create training and test data
training_data = SpeckDataset(5, 10**7)
test_data = SpeckDataset(5, 10**5)

# Creating the DataLoaders
batch_size = 5000

train_dataloader = DataLoader(training_data, batch_size)
test_dataloader = DataLoader(test_data, 5000)

print("Setting up neural network")
print("---" * 30)

# Checking if gpu is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))


model = NeuralNetwork().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X.float())
        loss = loss_fn(pred.float().reshape((-1,)), y.float())

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch % 100) == 0:
            loss, current = loss.item(), batch*len(X)
            print("Loss: {:>7f}   [{:>5d}/{:>5d}]".format(loss, current, size))

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X.float())
            test_loss += loss_fn(pred.float().reshape((-1,)), y.float()).item()
            # correct += (pred.argmax(1)==y).type(torch.float).sum().item()
            correct += (pred.round()==y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size*num_batches
    print("Test Error: \nAccuracy: {:>0.1f}, Avg loss: {:>8f} \n".format(100*correct, test_loss))

epochs = 5
for t in range(epochs):
    print("Epoch {}: \n".format(t+1) + "--"*30)
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

    


