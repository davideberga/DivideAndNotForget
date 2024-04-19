


import os
import re

import numpy as np

if __name__ == "__main__":
    # Get the current directory
    current_directory = os.path.join(os.getcwd(), 'results')

    datasets = ['cifar100', 'food101']
    methods = ['BASELINE', 'SEED']

    folder_in_directory = []
    stdout_in_directory = []

    # create list of the directory paths

    for path, subdirs, files in os.walk(current_directory):
        for name in files:
            if 'stdout' in name:
                stdout_in_directory.append(os.path.join(path, name))
                folder_in_directory.append(path)

    pattern = 'Epoch:.*Train loss.*Val loss.*Train acc.*Val acc: ([+-]?([0-9]*[.])?[0-9]+)'

    for i, file in enumerate(stdout_in_directory):
        with open(file) as f:
            task_accuracies = []
            task_accuracy = []
            row = f.readline()
            while row != '':
                match = re.match(pattern, row)
                if match is None and len(task_accuracy) > 0:
                    task_accuracies.append(task_accuracy)
                    task_accuracy = []
                elif match is not None:
                    task_accuracy.append(float(match.group(1)))
                row = f.readline()

        task_accuracies = np.round(np.array(task_accuracies), decimals=2)

        np.savetxt(os.path.join(folder_in_directory[i], 'results', 'task_valid_accs.txt'), task_accuracies/100, fmt='%.2f', delimiter='\t')

