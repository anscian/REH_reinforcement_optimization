from glob import glob
from tqdm import tqdm
from os import path, chdir
import csv
from itertools import chain

BASE_DIR = path.dirname(__file__)
BASE_DIR = path.join(BASE_DIR, 'dataset')
chdir(BASE_DIR)

BATCHES_DIR  = 'batches'
IMAGES_DIR   = 'images'
RESULTS_FILE = 'results.csv'

batches = glob(path.join(BASE_DIR, BATCHES_DIR, '*'))

for batch_path in tqdm(batches, desc='Combining Batch Images', unit='batches'):
    for imgfile in glob(path.join(batch_path, IMAGES_DIR, '*')):
        imgfile_new = path.join(BASE_DIR, IMAGES_DIR, path.basename(imgfile))
        with open(imgfile_new, 'wb') as fw:
            with open(imgfile, 'rb') as fr:
                fw.write(fr.read())

print('Compiling all results csv into one')
fieldnames = ['experiment', 'E', 'v', 'n', 'r', 'VF', 'frac_E']
result_files = [
    open(path.join(batch_path, RESULTS_FILE), 'r') 
    for batch_path in batches
]
data = chain(*[csv.DictReader(f) for f in result_files])

with open(path.join(BASE_DIR, RESULTS_FILE), 'w') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

for f in result_files:
    f.close()
