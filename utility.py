import os
import torch
import torch.utils.data as data

def sanitize(start_file_name: str = "cifar_net_5"):
    files_to_remove = [f for f in os.listdir() if os.path.isfile(f) and f.startswith(start_file_name)]
    for file in files_to_remove:
        os.remove(file)

def force_labels_classes_to_0N(labels, n_classes, desired_classes):
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

