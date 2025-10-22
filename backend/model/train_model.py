from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Current device: ", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root='./model/data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./model/data', train=False, download=True, transform=transform)

train_dataloader = DataLoader(train_data, 64, True)
test_dataloader = DataLoader(test_data, 1000)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x,2)
        x = torch.flatten(x,1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 5

for epoch in range(epochs):
    
    train_loss = 0
    test_loss = 0
    correct = 0

    model.train()

    for data, target in train_dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()

    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    train_loss /= len(train_dataloader)
    test_loss /= len(test_dataloader)
    accuracy = 100.0 * correct / len(test_data)

    print(
        f"Epoch {epoch+1}/{epochs}: "
        f"Train loss = {train_loss:.4f}, "
        f"Test loss = {test_loss:.4f}, "
        f"Accuracy = {accuracy:.2f}%"
    )

torch.save(model.state_dict(), "./model/classification_model.pth")