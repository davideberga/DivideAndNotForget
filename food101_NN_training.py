import os

import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torch.optim as optim
import torchvision

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



def get_data_for_classes(desired_classes, dataset):
    # Convert desired_classes into a tensor for broadcasting
    
    desired_classes_tensor = torch.tensor(desired_classes)[:, None]
    # Create a boolean mask indicating whether each target is in desired_classes
    mask = (torch.tensor(dataset._labels) == desired_classes_tensor).any(dim=0)

    # Extract indices where the mask is True
    indices = torch.nonzero(mask).squeeze().tolist()

    filtered_data = torch.utils.data.Subset(dataset, indices)

    # Create the DataLoader with the filtered dataset
    return data.DataLoader(filtered_data, batch_size=8, shuffle=True, num_workers=8)


def cifar_train(resnet50):
    PATH_TRAIN = './food101_net_5_CLASSES'

    def no_grad(nn_part = None):
        if(nn_part != None):
            for param in nn_part.parameters():
                param.requires_grad = False

    # Remove gradients tracking on convnet and avgpool
    #no_grad(resnet50.features)
    #no_grad(resnet50.avgpool)

    resnet50.classifier[6] = nn.Identity()
    resnet50.classifier[6] = nn.Linear(4096, 5)

    if(os.path.exists(PATH_TRAIN)):
        try:
            resnet50.load_state_dict(torch.load(PATH_TRAIN))
            print(f'Training resumed from {PATH_TRAIN}')
        except Exception as e:
            print(e)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=resnet50.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    
    EPOCHS = 1000
    last_saved = ""
    total_steps = 1
    for epoch in range(EPOCHS):
        resnet50.train()
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
            if i % 100 == 0:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item()}')
            if total_steps % 1000 == 0:
                sanitize()
                last_saved = PATH_TRAIN + f'__{epoch}__epoch__{i}__batch.pth'
                torch.save(resnet50.state_dict(), last_saved)
                cifar_test(resnet50, last_saved)
            total_steps += 1

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

    PATH_TEST = './food101_net_5_CLASSES__1__epoch__700__batch.pth'

    food101_train = datasets.Food101(root='./data', split='train', download=True, transform=transform)
    food101_test = datasets.Food101(root='./data', split='test', download=True, transform=transform)


    # Filter the dataset to only include classes 0-4
    desired_classes = [0, 1, 2, 3, 5 ]
    train_loader = get_data_for_classes(desired_classes, food101_train)
    test_loader = get_data_for_classes(desired_classes,  food101_test)

    resnet50 = torchvision.models.alexnet().to(DEVICE)

    cifar_train(resnet50)
    # cifar_test(resnet50,PATH_TEST)





