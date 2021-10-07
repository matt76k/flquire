import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
import util
from torch._C import device

import model


def test(net, device):
    net.eval()

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_dataset = datasets.CIFAR10(
        "data/cifar", train=False, download=True, transform=test_transform
    )

    data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )

    correct = 0
    total = 0

    for imgs, labels in tqdm.tqdm(data_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = net(imgs)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels.flatten()).sum().item()

    return correct / total


net = model.Net()
device = "cuda" if torch.cuda.is_available() else "cpu"

net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
criterion = torch.nn.CrossEntropyLoss().to(device)

train_transform = transforms.Compose(
    [
        util.Cutout(num_cutouts=2, size=8, p=0.8),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_dataset = datasets.CIFAR10(
    "data/cifar", train=True, download=True, transform=train_transform
)
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

best = 0

for epoch in range(100):
    print(f"epoch {epoch}")
    net.train()

    for imgs, labels in tqdm.tqdm(data_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(imgs)
        loss = criterion(outputs, labels.squeeze_())
        loss.backward()
        optimizer.step()

    acc = test(net, device)
    print(f"acc = {acc}")

    if best < acc:
        best = acc
        net.cpu()
        torch.save(net.state_dict(), "res9.pth")
        net.to(device)
