import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import torch
import pygad
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision
import torchvision.models as models



from utility import get_data_for_classes, sanitize, force_labels_classes_to_0N




def on_gen(ga_instance):
    global resnet50
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])

    total = correct = 0.0

    resnet50.eval()

    solution, solution_fitness, _ = ga_instance.best_solution()

    with torch.no_grad():
        best_weights_tensor = torch.tensor(solution, dtype=torch.float32).view(1, NUM_GENES)
        resnet50.classifier[6].weight.data = best_weights_tensor

        for data in test_loader:
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # SET label 0 to class 1, all others lables to class 0
            labels[labels > 0] = 1
            torch.logical_not(labels).to(torch.float32)

            resnet50 = resnet50.to(DEVICE)
            # calculate outputs by running images through the network
            outputs = resnet50(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network: {(correct / total):.5f} %')



    torch.save(resnet50.state_dict(),'best_alexnet_1out_weights.pth')


# 2. Define the fitness function for PyGAD
def fitness_func(modelx, solution, solution_idx):
    total = correct = 0.0

    i = 0
    global resnet50

    with torch.no_grad():
        for data in test_loader:
            resnet50.eval()

            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # SET label 0 to class 1, all others lables to class 0
            labels[labels > 0] = 1
            torch.logical_not(labels).to(torch.float32)

            weights_tensor = torch.tensor(solution, dtype=torch.float32).view(1, NUM_GENES)
            resnet50.classifier[6].weight.data = weights_tensor

            resnet50 = resnet50.to(DEVICE)
            # calculate outputs by running images through the network
            outputs = resnet50(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            loss = torch.nn.functional.mse_loss(outputs, labels).item()
            i += 1
            # if i % 5 == 0:
            #     print(f'i:{i}\t FITNESS:{-loss}')
    # For PyGAD, we want to maximize the fitness, so we will take the negative of the loss
    return -loss



if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    NUM_GENES = 4096

    new_classifier = nn.Linear(NUM_GENES, 1)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize RESNET50
    resnet50 = torchvision.models.alexnet(torchvision.models.AlexNet_Weights.IMAGENET1K_V1).to(DEVICE)

    food101_train = datasets.Food101(root='./data', split='train', download=True, transform=transform)
    food101_test = datasets.Food101(root='./data', split='test', download=True, transform=transform)

    # Filter the dataset to only include classes 0-4
    desired_classes = [0, 1]#, 2, 3, 5]
    train_loader = get_data_for_classes(desired_classes, food101_train)
    test_loader = get_data_for_classes(desired_classes, food101_test)

    classifier_weights = resnet50.classifier[6].weight.data

    resnet50.classifier[6] = nn.Identity()  # Removing the last layer
    resnet50.classifier[6] = new_classifier

    for param in resnet50.parameters():
        param.requires_grad = False

    mean_weights = classifier_weights.mean(dim=0)
    std_weights = torch.std(classifier_weights, dim=0).cpu().numpy()

    mean_max_class = 1
    mean_min_class = 1
    # mean_max_class = torch.mean(max_class).to(DEVICE)
    # mean_max_class = torch.mean(min_class).to(DEVICE)

    N_CLASSES = 1


    # Number of solutions (weights sets) in the population
    sol_per_pop = 200


    # Initialize PyGAD parameters
    num_generations = 10
    num_parents_mating = 5

    # Initial population of possible solutions
    init_range_low = -std_weights
    init_range_high = std_weights

    PERTURBATION_RANGE = 0.005  # adjust this as needed

    initial_population = np.random.uniform(low=init_range_low, high=init_range_high, size=(sol_per_pop, NUM_GENES))

    # initial_population = mean_weights.unsqueeze(0) + (torch.rand((100, NUM_GENES)) - 0.5) * 2 * PERTURBATION_RANGE

    fitness_function = fitness_func

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=NUM_GENES,
        init_range_low=init_range_low,
        init_range_high=init_range_high,
        parent_selection_type="sss",
        keep_parents=1,
        crossover_type="two_points",
        crossover_probability=0.8,
        mutation_type="swap",
        mutation_probability=0.8,
        gene_space= {"low": -std_weights*1.2, "high": std_weights*1.2},
        initial_population=initial_population,
        on_generation=on_gen
    )

    # Run the GA
    ga_instance.run()

    # Get the best solution
    solution, solution_fitness, _ = ga_instance.best_solution()
    print("Best solution fitness:", solution_fitness)




    # PLOT @@@
    #ga_instance.plot_fitness()
    # PLOT @@@

