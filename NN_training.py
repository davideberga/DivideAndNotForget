import os

import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torch.optim as optim

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def sanitize(start_file_name : str = "cifar_net_5"):
    files_to_remove = [f for f in os.listdir() if os.path.isfile(f) and f.startswith(start_file_name)]
    for file in files_to_remove:
        os.remove(file)



def get_data_for_classes(desired_classes,original_targets_tensor):
    # Convert desired_classes into a tensor for broadcasting
    desired_classes_tensor = torch.tensor(desired_classes)[:, None]
    # Create a boolean mask indicating whether each target is in desired_classes
    mask = (original_targets_tensor == desired_classes_tensor).any(dim=0)

    # Extract indices where the mask is True
    indices = torch.nonzero(mask).squeeze().tolist()

    filtered_data = torch.utils.data.Subset(cifar10_train, indices)

    # Create the DataLoader with the filtered dataset
    return data.DataLoader(filtered_data, batch_size=32, shuffle=True, num_workers=32)


def cifar_train(resnet50):
    PATH_TRAIN = './cifar_net_5_CLASSES'

    for param in resnet50.parameters():
        param.requires_grad = False
    try:
        # if exists try to load
        resnet50.fc = nn.Identity()
        resnet50.fc = nn.Linear(2048, 5)
        resnet50.load_state_dict(torch.load(PATH_TRAIN))
    except:
        # if not exists, start from scratch
        resnet50.fc = nn.Identity()
        resnet50.fc = nn.Linear(2048,5)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet50.fc.parameters(), lr=0.001)

    resnet50.train()
    EPOCHS = 1000
    last_saved = ""
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            resnet50 = resnet50.to(DEVICE)
            labels = labels.to(DEVICE)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = resnet50(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item()}')
                running_loss = 0.0

                sanitize()
                last_saved = PATH_TRAIN + f'__{epoch}__epoch__{i}__batch.pth'
                torch.save(resnet50.state_dict(), last_saved)
        cifar_test(resnet50, last_saved)

def cifar_test(resnet50, PATH_TEST):

    resnet50.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            resnet50 = resnet50.to(DEVICE)
            # calculate outputs by running images through the network
            outputs = resnet50(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {(correct / total):.5f} %')



if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    PATH_TEST = './cifar_net_5_CLASSES__1__epoch__700__batch.pth'

    cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # Filter the dataset to only include classes 0-4
    desired_classes = [0, 1, 2, 3, 4]
    original_targets_train = torch.tensor(cifar10_train.targets)
    original_targets_test = torch.tensor(cifar10_test.targets)
    train_loader = get_data_for_classes(desired_classes,original_targets_train)
    test_loader = get_data_for_classes(desired_classes,original_targets_test)

    resnet50 = models.resnet50().to(DEVICE)

    cifar_train(resnet50)
    # cifar_test(resnet50,PATH_TEST)





