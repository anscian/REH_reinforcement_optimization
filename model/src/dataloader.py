from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import polars as pl
from glob import glob
from os import path


pil_to_tensor = transforms.PILToTensor()


class ReinforcementDataset(Dataset):
    def __init__(self, img_dir : str, img_ext : str, data : pl.LazyFrame = None, transform : callable = None) -> None:
        '''
        Initializes a dataloader for our REH reinforcement dataset
        ---------------------------------------------------------------------------------------------------
        Parameters:
            img_dir   : Path to the directory containing all the images
            img_ext   : Image extension for each image (assumed file path {img_dir}/{experiment}.{img_ext})
            data      : polars.LazyFrame containing mapping to target and scalers
            transform : Transformation logic for the loaded image to preprocess it (ignored if None)
        Returns:
        '''

        assert data is not None, 'Data passed can\'t be NoneType'

        self.img_dir = img_dir
        self.img_ext = ('' if img_ext.startswith('.') else '.') + img_ext
        self.transform = transform
        
        self.data = data
        self.experiments = self.data.collect()['experiment'].to_list()


    def __len__(self):
        return len(self.experiments)


    def __getitem__(self, idx : int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        The image gets loaded first as PIL.Image, converted to grayscale and then to torch.Tensor
        The dimensions of each image tensor is (1, H, L)
        '''

        experiment = self.experiments[idx]

        img = Image.open(path.join(self.img_dir, experiment + self.img_ext)).convert('L')
        if self.transform is not None: img = self.transform(img)
        img = pil_to_tensor(img).float()

        results = self.data.filter(pl.col('experiment') == experiment).collect()
        scaler  = results.select('n', 'r', 'VF').to_torch().flatten().float()
        target  = results.select('frac_E', 'v').to_torch().flatten().float()

        return img, scaler, target


def load_data(data_path : str, *args, train_test_split : bool = True, **kwargs) -> tuple[ReinforcementDataset, ReinforcementDataset] | ReinforcementDataset:
    '''
    Provides entire dataset either completely or as test train split
    --------------------------------------------------------------------------------------------------------
    Parameters:
        data_path        : Path to the csv file containing mapping to target and scalers
        train_test_split : Should be splitted or not, test set is single entry for each (n, r) [n > 0]
        args             : all the positional arguments for ReinforcementDataset initializer (see docs)
        kwargs           : all keyword arguments for ReinforcementDataset initializer except data (see docs)
    Returns:
        Either tuple of two datasets (train, test) or complete dataset as per `train_test_split`
    '''

    data = pl.scan_csv(data_path)
    make_dataset = lambda x : ReinforcementDataset(*args, data=x, **kwargs)

    if train_test_split:
        test  = data.group_by(['n', 'r']).first().filter(pl.col('n') > 0)
        train = data.join(test, on='experiment', how='anti')

        return tuple(map(make_dataset, (train, test)))
    else:
        return make_dataset(data)
