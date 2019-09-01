import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train', order=['A','B','C']):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.domain_num = len(order)

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/%s' % (mode, order[0])) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/%s' % (mode, order[1])) + '/*.*'))

        if self.domain_num == 3:
            self.files_C = sorted(glob.glob(os.path.join(root, '%s/%s' % (mode, order[2])) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB') )

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB') )
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB') )

        if self.domain_num == 2:
            return {'A': item_A, 'B': item_B}
        else :
            if self.unaligned:
                item_C = self.transform(Image.open(self.files_C[random.randint(0, len(self.files_C) - 1)]).convert('RGB') )
            else:
                item_C = self.transform(Image.open(self.files_C[index % len(self.files_C)]).convert('RGB') )

            return {'A': item_A, 'B': item_B, 'C': item_C}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B)) #, len(self.files_C))