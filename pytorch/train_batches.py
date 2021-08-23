import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.DataLoader import SpeckDataset
from models.model import NeuralNetwork



print("Loading Data...")
print("--" * 30)

# Create training and test data
training_data = SpeckDataset(5, 10**6, 5)
test_data = SpeckDataset(5, 10**5)

# Creating the DataLoaders
batch_size = 500

train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, 500, shuffle=True)

print("Initializing neural network")
print("--" * 30)

# Checking if gpu is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))


model = NeuralNetwork().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 1e-4, 1e-3, cycle_momentum=False)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        predictions = torch.empty((y.shape[0],X.shape[-1]))

        # Compute prediction error
        for group in range(5):
            predictions[:,group] = model(X[:,:,:,group].float())

        
        prediction_avg = predictions.mean(axis=1).to(device)
        
        loss = loss_fn(prediction_avg.float().reshape((-1,)), y.float())

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (batch % 100) == 0:
            loss, current = loss.item(), batch*len(X)
            print("Loss: {:>7f}   [{:d}/{:d}]".format(loss, current//batch_size, size//batch_size))

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
            correct += torch.eq(torch.ge(pred, 0.5), y).sum().item()

    test_loss /= num_batches
    correct /= size
    print("Test Error: \nAccuracy: {:>0.1f}%, Avg loss: {:>8f} \n".format(100*correct, test_loss))

epochs = 200
for t in range(epochs):
    print("Epoch {}: \n".format(t+1) + "--"*30)
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

    


