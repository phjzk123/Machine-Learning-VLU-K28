import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import SGD
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms

# Kiểm tra thiết bị
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Chuẩn hóa dữ liệu với mean = 0.5 và std = 0.5
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Tải dữ liệu CIFAR10 và chia thành train/validation
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_size = int(0.9* len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Tạo DataLoader cho tập train và validation
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)

# Tải tập test
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# Hiển thị 5 ảnh đầu tiên trong tập dữ liệu test
def imshow(img):
    img = img / 2 + 0.5  # Hoàn nguyên quá trình chuẩn hóa
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(testloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images[:5]))

# Xây dựng mô hình MLP
def getModel(n_features):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_features, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

# Khởi tạo model, loss function và optimizer
n_features = 32 * 32 * 3  # CIFAR-10 có kích thước ảnh là 32x32 với 3 kênh màu (RGB)
model = getModel(n_features).to(device)
lr = 0.01
optim = SGD(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# Đánh giá model
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy

# Training và đánh giá model
n_epochs = 100
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for epoch in range(n_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optim.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optim.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    train_loss = running_loss / len(trainloader)
    val_loss, val_accuracy = evaluate(model, valloader, loss_fn)
    print(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

# Vẽ đồ thị Loss và Accuracy
plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.title('Loss per Epoch')
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.subplot(122)
plt.title('Accuracy per Epoch')
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.legend()
plt.show()
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import SGD
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Chuẩn hóa dữ liệu với mean và std
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Tải dữ liệu CIFAR10
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

def getModel(n_features):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),  # Thêm Dropout
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.5),  # Thêm Dropout
        nn.Linear(64, 10)
    )
    return model

n_features = 32 * 32 * 3
model = getModel(n_features).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.1)
loss_fn = nn.CrossEntropyLoss()

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy

n_epochs = 100
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for epoch in range(n_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optim.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optim.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()  # Điều chỉnh tốc độ học
    train_accuracy = 100 * correct / total
    train_loss = running_loss / len(trainloader)
    val_loss, val_accuracy = evaluate(model, valloader, loss_fn)
    print(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.title('Loss per Epoch')
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.subplot(122)
plt.title('Accuracy per Epoch')
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.legend()
plt.show()