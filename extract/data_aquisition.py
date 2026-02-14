import json
import os
import sys
import numpy as np
import csv

COMP_LIST = ['dataset', 'config.json', 'data_aquisition.py', 'reh8x8.py']
def clear():
    try:
        [os.remove(f) for f in os.listdir() if f not in COMP_LIST]
        return 0
    except:
        return 1

def experiment(rein_params : dict = {}):
    with open('config.json', 'w') as f:
        json.dump(rein_params, f)
    os.system('abaqus cae noGUI=reh8x8.py')
    clear()


if __name__ == '__main__':
    with open('./dataset/results.csv', 'w', newline='') as f:
        csv_writer = csv.DictWriter(f, fieldnames=['experiment', 'E', 'v', 'n', 'r', 'VF', 'frac_E'])
        csv_writer.writeheader()
    [os.remove(f'./dataset/images/{f}') for f in os.listdir('./dataset/images')]
    clear()

    # N_MAX_VAR, RXTRA_VAR, RUNS = tuple(map(int, sys.argv[1:])), (0.0, 8.2, 0.2), 8
    N_MAX_VAR, RXTRA_VAR, RUNS = (20, 21, 5), (1.0, 2.0, 0.5), 1

    print('Base Experiment')
    experiment({})

    for n_max in range(*N_MAX_VAR):
        for rxtra in np.arange(*RXTRA_VAR):
            exp = dict(n_max = n_max, rxtra = rxtra)
            s = f'\nExperiment {", ".join([" = ".join(map(str, x)) for x in exp.items()])}'
            print(s, '*' * len(s), sep='\n')
            for i in range(1, RUNS + 1):
                print('Run', i)
                experiment(dict(n_max = n_max, rxtra = rxtra))
