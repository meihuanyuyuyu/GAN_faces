import os
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import torchvision.transforms as transform
from torch.utils.data import Dataset



class crypkodata(Dataset):
    def __init__(self,dir:str,transforms) -> None:
        super().__init__()
        self.fp = os.path.join(os.getcwd(),dir)
        self.transforms = transforms
        self.data = os.listdir(self.fp)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.fp,self.data[index]))
        img= self.transforms(img)
        return img
    
    def __len__(self):
        return len(self.data)
