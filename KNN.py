import os

import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import torchvision.datasets as datasets

import torch.utils.data as data
import torch.optim as optim
import torchvision

from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

N_SVD_FEATURES = 50

import torch

class KNN:
    def __init__(self):
        self.train_data = torch.empty(0,N_SVD_FEATURES).to(DEVICE)
        self.train_labels = torch.empty(0,1).to(DEVICE)

    def fit(self, train_data, train_labels):
        self.train_data = torch.cat((self.train_data, train_data),dim=0)
        self.train_labels = torch.cat((self.train_labels, train_labels.unsqueeze(1)),dim=0)

    def set_k(self, k):
        self.k = k

    def predict(self, test_data):
        # Ensure the data is float
        test_data = test_data.float()
        self.train_data = self.train_data.float()

        # Compute pairwise distances
        dists = torch.cdist(test_data, self.train_data)

        # Get the k smallest distances and their indices
        _, indices = dists.topk(self.k, dim=1, largest=False)

        # Use indexing to get the labels of the k-nearest neighbors
        knn_labels = self.train_labels[indices]

        # Vote for the most common label
        predictions, _ = knn_labels.mode(dim=1)

        return predictions

def svd(data, n_features):


    # PCA
    # Center the data
    mean = torch.mean(data, dim=0)
    centered_data = data - mean

    # Compute the covariance matrix
    covariance_matrix = torch.mm(centered_data.t(), centered_data) / centered_data.size(0)

    # Compute the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
    eigenvalues = eigenvalues.real
    # Convert eigenvectors to real and cast to float32
    eigenvectors_real = eigenvectors.real

    # Sort the eigenvectors by descending eigenvalues
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    topk_eigenvectors = eigenvectors_real[:, sorted_indices[:n_features]]

    # Transform the data to the reduced-dimensional space
    transformed_data = torch.mm(centered_data, topk_eigenvectors.real)

    return transformed_data
    # PCA


    # SVD
    # n_rows = data.shape[0]
    #
    # # Center the data
    # mean = torch.mean(data, dim=0)
    # centered_data = data - mean
    #
    # # Compute the SVD
    # U, S, V = torch.svd(centered_data)
    #
    # # Determine the number of features to retain
    # actual_features = min(n_features, U.shape[1])
    #
    # # Retain only the top actual_features singular vectors/values
    # U_k = U[:, :actual_features]
    # S_k = torch.diag(S[:actual_features])
    #
    # # Compute the reduced representation
    # transformed_data = torch.mm(U_k, S_k)
    #
    # # If actual_features is less than n_features, pad with zeros
    # if actual_features < n_features:
    #     padding = torch.zeros((n_rows, n_features - actual_features), device=data.device)
    #     transformed_data = torch.cat([transformed_data, padding], dim=1)
    #
    # return transformed_data
    # SVD



def force_labels_classes_to_0N(labels, n_classes):
    tensor_desired_classes = torch.tensor(desired_classes)
    # Create a dictionary to map the sorted unique numbers to 0 to 4
    mapping = {tensor_desired_classes[i].item(): i for i in range(n_classes)}
    # Replace elements in the tensor using the mapping
    new_tensor = torch.tensor([mapping[x.item()] for x in labels])

    return new_tensor


def get_data_for_classes(desired_classes, dataset):
    # Convert desired_classes into a tensor for broadcasting

    desired_classes_tensor = torch.tensor(desired_classes)[:, None]
    # Create a boolean mask indicating whether each target is in desired_classes
    mask = (torch.tensor(dataset._labels) == desired_classes_tensor).any(dim=0)

    # Extract indices where the mask is True
    indices = torch.nonzero(mask).squeeze().tolist()

    filtered_data = torch.utils.data.Subset(dataset, indices)

    # Create the DataLoader with the filtered dataset
    return data.DataLoader(filtered_data, batch_size=32, shuffle=True, num_workers=12)


def knn_space_creation(resnet50):


    # Initialize the model
    model = KNN()

    for param in resnet50.parameters():
        param.requires_grad = False

    resnet50.classifier[6] = nn.Identity()

    resnet50.eval()
    with torch.no_grad():
        for i_train, data_train in tqdm(enumerate(train_loader, 0)):

            inputs_train, labels_train = data_train
            labels_train = force_labels_classes_to_0N(labels_train, 5)

            # shuffle the inputs and labels in each batch
            perm = torch.randperm(inputs_train.shape[0])
            inputs_train = inputs_train[perm]
            labels_train = labels_train[perm]


            inputs_train = inputs_train.to(DEVICE)
            labels_train = labels_train.to(DEVICE)

            resnet50 = resnet50.to(DEVICE)


            outputs_train = resnet50(inputs_train)
            outputs_train = svd(outputs_train, n_features=N_SVD_FEATURES)


            # Fit to training data
            model.fit(outputs_train, labels_train)

    return model


def knn_computation(resnet50, model, k):
        accuracies = torch.empty(0,1).to(DEVICE)

        model.set_k(k)

        for i_test, data_test in enumerate(test_loader, 0):
            inputs_test, labels_test = data_test
            labels_test = force_labels_classes_to_0N(labels_test, 5)

            inputs_test = inputs_test.to(DEVICE)
            labels_test = labels_test.to(DEVICE)

            outputs_test = resnet50(inputs_test)

            outputs_test = svd(outputs_test, n_features=N_SVD_FEATURES)

            # Predict
            predictions = model.predict(outputs_test)

            accuracy = (predictions == labels_test).float().mean().item()


            accuracies = torch.cat((accuracies,torch.tensor(accuracy,device=DEVICE).reshape(1,1)))

        print(f'K: {k}\tAccuracy: {torch.mean(accuracies)}')


if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    food101_train = datasets.Food101(root='./data', split='train', download=True, transform=transform)
    food101_test = datasets.Food101(root='./data', split='test', download=True, transform=transform)

    # Filter the dataset to only include classes 0-4
    desired_classes = [0, 1, 2, 3, 5]
    train_loader = get_data_for_classes(desired_classes, food101_train)
    test_loader = get_data_for_classes(desired_classes, food101_test)

    resnet50 = torchvision.models.alexnet(torchvision.models.AlexNet_Weights.IMAGENET1K_V1).to(DEVICE)

    knn_model = knn_space_creation(resnet50)

    for k in range(30,100,5):
        knn_computation(resnet50, knn_model, k)





