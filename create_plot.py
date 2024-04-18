

import os
import re
import shutil

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score

def plot_accuracy_or_loss(current_directory, paths, type_plot, dataset, method):

    if type_plot == 'losses':
        type_plot_name = 'Loss'
    elif type_plot == 'accs':
        type_plot_name = 'Accuracy'


    colors = [('red','lightcoral'),('blue', 'navy'), ('green', 'lime')]

    path_file = os.path.join(current_directory, paths[0], 'results', f'task_train_{type_plot}.txt')
    file = np.loadtxt(path_file)
    number_tasks = file.shape[0]
    number_epochs = file.shape[1] * number_tasks

    custom_ticks = [25 + tick * (number_epochs/number_tasks) for tick in range(number_tasks)]
    custom_labels = [ f'T{tick+1}' for tick in range(number_tasks)]

    fig, ax = plt.subplots()

    max_value = 0

    for i, path in enumerate(paths):
        model_name = path.split('_')[-2]

        path_train_losses = os.path.join(current_directory, path, 'results', f'task_train_{type_plot}.txt')
        path_valid_losses = os.path.join(current_directory, path, 'results', f'task_valid_{type_plot}.txt')


        train_losses = np.loadtxt(path_train_losses).flatten()
        valid_losses = np.loadtxt(path_valid_losses).flatten()

        max_value = max_value if max_value > np.max(train_losses) else np.max(train_losses)
        max_value = max_value if max_value > np.max(valid_losses) else np.max(valid_losses)

        ax.plot(train_losses, label=f'Train {type_plot_name} {model_name}', color=colors[i][0])
        ax.plot(valid_losses, label=f'Valid {type_plot_name} {model_name}', color=colors[i][1])

    plt.ylim(bottom=0)

    y_label = (15/100) * max_value
    for i, label in zip(custom_ticks, custom_labels):
        ax.text(i, -y_label, label, ha='center')

    plt.title(f'Train and Validation {type_plot_name}')
    plt.xticks(np.arange(1,number_epochs+1,number_epochs/number_tasks))
    plt.tight_layout()
    plt.legend()
    # plt.show()
    plt.savefig(f'{current_directory}/{dataset}_{method}/train_valid_{type_plot_name}.png')
    plt.clf()

def plot_test_accuracy_aware(current_directory, paths, dataset, method):
    # AWARE PLOT
    # Plot last row (plur or minus the value is the same for each row)

    path_file = os.path.join(current_directory, paths[0], 'results', f'acc_taw.txt')
    file = np.loadtxt(path_file)
    number_tasks = file.shape[0]

    colors = ['red', 'blue', 'green']

    for i, path in enumerate(paths):
        model_name = path.split('_')[-2]

        aware_path = os.path.join(current_directory, path, 'results', 'acc_taw.txt')
        with open(aware_path, 'r') as accuracy_aware_file:
            row = accuracy_aware_file.readline()
            final_accuracy = None
            n_row = 0
            while row != '':
                n_row += 1
                if n_row == number_tasks:
                    final_accuracy = row.split('\t')
                row = accuracy_aware_file.readline()

        final_accuracy = np.round(np.array(final_accuracy).astype(float), decimals=4)
        plt.plot(final_accuracy, label=f'Accuracy Aware {model_name}', color=colors[i])

    plt.xticks(np.arange(1,number_tasks,1))
    plt.title('Test Accuracy Aware')
    plt.xlabel('TASK')
    plt.ylabel('ACCURACY')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{current_directory}/{dataset}_{method}/test_accuracy_aware.png')
    plt.clf()

def plot_test_accuracy_agnostic(current_directory, paths, dataset, method):
    # AGNOSTIC PLOT
    # Plot diagonal + plot last row (performance on model complete after all tasks)

    path_file = os.path.join(current_directory, paths[0], 'results', f'acc_tag.txt')
    file = np.loadtxt(path_file)
    number_tasks = file.shape[0]


    colors = [('red','lightcoral'),('blue', 'navy'), ('green', 'lime')]


    for i, path in enumerate(paths):
        model_name = path.split('_')[-2]

        agnostic_path = os.path.join(current_directory, path, 'results', 'acc_tag.txt')
        with open(agnostic_path, 'r') as accuracy_agnostic_file:
            row = accuracy_agnostic_file.readline()
            accuracy_by_task = []
            final_accuracy = None
            n_row = 0
            while row != '':
                accuracy_by_task.append(row.split('\t')[n_row])
                n_row += 1
                if n_row == number_tasks:
                    final_accuracy = row.split('\t')
                row = accuracy_agnostic_file.readline()


        accuracy_by_task = np.round(np.array(accuracy_by_task).astype(float), decimals=4)
        final_accuracy = np.round(np.array(final_accuracy).astype(float), decimals=4)

        plt.plot(accuracy_by_task, label=f'Accuracy after each task {model_name}', color=colors[i][0])
        plt.plot(final_accuracy, label=f'Accuracy after training {model_name}', color=colors[i][1])

    plt.xticks(np.arange(1,number_tasks,1))
    plt.title('Accuracy Agnostic')
    plt.xlabel('TASK')
    plt.ylabel('ACCURACY')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{current_directory}/{dataset}_{method}/test_accuracy_agnostic.png')
    plt.clf()


def plot_test_precision_recall(current_directory, paths, dataset, method):
    # PRECISION AND RECALL

    path_file = os.path.join(current_directory, paths[0], 'results', f'acc_tag.txt')
    file = np.loadtxt(path_file)
    number_tasks = file.shape[0]

    colors = [('red','lightcoral'),('blue', 'navy'), ('green', 'lime')]

    for i, path in enumerate(paths):
        model_name = path.split('_')[-2]

        precision_agnostic = []
        recall_agnostic = []

        for task in range(number_tasks):
            predictions_agnostic_path = os.path.join(current_directory, path, 'results',
                                                     f'task_{task}_tag_pred_complete.txt')
            targets_path = os.path.join(current_directory, path, 'results',
                                        f'task_{task}_targets_complete.txt')

            # Read predictions and target labels from files
            predictions_agnostic = np.loadtxt(predictions_agnostic_path)
            targets = np.loadtxt(targets_path)

            # Compute precision
            precision_agnostic.append(precision_score(targets, predictions_agnostic, average='weighted'))

            # Compute recall
            recall_agnostic.append(recall_score(targets, predictions_agnostic, average='weighted'))

        precision_agnostic = np.round(np.array(precision_agnostic).astype(float), decimals=4)
        recall_agnostic = np.round(np.array(recall_agnostic).astype(float), decimals=4)

        plt.plot(precision_agnostic, label=f'Precision {model_name}', color=colors[i][0])
        plt.plot(recall_agnostic, label=f'Recall {model_name}', color=colors[i][1])

    plt.xticks(np.arange(1,number_tasks,1))
    plt.title('Precision-Recall Agnostic')
    plt.xlabel('TASK')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{current_directory}/{dataset}_{method}/test_precision_recall.png')
    plt.clf()

def save_plot(current_directory, paths, dataset, method):

    # sort the Resnet model 18,50,101
    paths.sort(key=lambda x: int(re.match(r'.*_res(\d+).*', x).group(1)))

    # LOSS
    plot_accuracy_or_loss(current_directory=current_directory, paths=paths, type_plot='losses',
                          dataset=dataset, method=method)

    # TRAIN/VAL ACCURACY AGNOSTIC
    plot_accuracy_or_loss(current_directory=current_directory, paths=paths, type_plot='accs',
                          dataset=dataset, method=method)

    # TEST ACCURACY AWARE
    plot_test_accuracy_aware(current_directory=current_directory, paths=paths,
                          dataset=dataset, method=method)

    # TEST ACCURACY AGNOSTIC
    plot_test_accuracy_agnostic(current_directory=current_directory, paths=paths,
                          dataset=dataset, method=method)

    # TEST PRECISIOM-RECALL
    plot_test_precision_recall(current_directory=current_directory, paths=paths,
                          dataset=dataset, method=method)


if __name__ == "__main__":
    # Get the current directory
    current_directory = os.path.join(os.getcwd(), 'results')


    datasets = ['cifar100', 'food101']
    methods = ['BASELINE', 'SEED']

    folder_in_directory = []

    # create list of the directory paths
    for i, folder in enumerate(os.listdir(current_directory)):
        if os.path.isdir(os.path.join(current_directory, folder)):
            folder_in_directory.append(folder)

    # iterate over datasets
    for dataset in datasets:
        paths = []
        # iterate over methods
        for method in methods:
            for dir in folder_in_directory:
                match = re.search(f'{dataset}.*{method}', dir)
                if match is not None:
                    paths.append(dir)
            current_method = method

        os.makedirs(os.path.join(current_directory, f'{dataset}_{method}'), exist_ok=True)
        save_plot(current_directory=current_directory, paths=paths, dataset=dataset, method=method)
