import torch
from functools import wraps


class Dataset(torch.utils.data.Dataset):
    """
    This is just the torch.utils.data.Dataset class. 
    """
    # Names of entries in data structures
    names_indices = {
        'mask' : 0,
        'HKL' : 1,
        'miller_indices' : 1, #alias
        'resolution' : 2,
        'dHKL' : 2, #alias
        'wavelength' : 3,
        'lambda' : 3, #alias
        'metadata' : 4, 
        'I' : 5,
        'intensities' : 5,
        'SIGI' : 6,
        'SigI' : 6, #alias
        'uncertainties' : 6, #alias
    }
    def __getitem__(self, idx):
        """
        Subclasses of abismal.io.Dataset must implement
        __getitem__ which returns a data structure containing
        (
            HKL : n x 3 int32 tensor,
            resolution : n x 1 float32 tensor,
            wavelength : n x 1 float32 tensor,
            metadata : n x d float32 tensor,
            I : n x 1 float32 tensor,
            SIGI : n x 1 float32 tensor,
        )
        """
        raise NotImplementedError(help(Dataset.__getitem__))

    @staticmethod
    def collate_fn(batch):
        """
        This method will collate a batch of reflection data from
        multiple images into a masked tensor. 
        """
        l = max([len(i[0]) for i in batch])
        out = []
        for image in batch:
            n = len(image[0])
            m = l - n
            mask = torch.ones_like(image[0][...,:1], dtype=torch.bool)
            mask = torch.nn.functional.pad(i, (0, 0 , 0, m))
            image = [torch.nn.functional.pad(i, (0, 0, 0, m)) for i in image]
            out.append([mask] + image)
            
        out = torch.stack(out)
        return out

    @staticmethod
    def get_item(data, key):
        pass

class DataLoader(torch.utils.data.DataLoader):
    @wraps(torch.utils.data.DataLoader) #copy dataloader docstring
    def __init__(self, *args, **kwargs):
        collate_fn = kwargs.pop("collate_fn", Dataset.collate_fn)
        super().__init__(*args, collate_fn=collate_fn, **kwargs)

