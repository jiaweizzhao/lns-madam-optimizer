# A toy example showing how to use LNS_Madam optimizer in PyTorch.

import torch
from torch.nn import Linear
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import mse_loss
from lns_madam import LNS_Madam


# create a toy regression dataset
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# create a PyTorch DataLoader for batching the dataset
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10)

# create a linear regression model
model = Linear(10, 1)

# create an instance of LNS_Madam optimizer
optimizer = LNS_Madam(model.parameters())

# train the model
num_epochs = 10
for epoch in range(num_epochs):
    for i, (batch_X, batch_y) in enumerate(dataloader):
        # forward pass
        preds = model(batch_X)
        loss = mse_loss(preds, batch_y)

        # backward pass and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
