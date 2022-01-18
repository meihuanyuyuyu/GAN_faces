from utils import cross_train, train_D,train_G, vision_results
from torch.utils.data.dataloader import DataLoader
import torch
from torch.optim import SGD,Adam
from torchvision.transforms.transforms import Normalize, CenterCrop, RandomHorizontalFlip, ToTensor
from utils.data import crypkodata
from torch.nn import BCELoss
from config import *
import torchvision.transforms as tranfom
import model
from tqdm import tqdm



if __name__ =='__main__':
    face_data = crypkodata('GAN/crypko_data/faces',transforms=tranfom.Compose([ToTensor(),CenterCrop(64),RandomHorizontalFlip(),Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]))
    r_data = DataLoader(face_data,batch_size=batch_size,shuffle=True,num_workers=4)
    netg = model.netG(noise_dim,is_bn=True)
    netd = model.netD(3)
    optimizer_g = Adam(netg.parameters(),lr=lr_g,weight_decay=1e-7)
    optimizer_d = Adam(netd.parameters(),lr=lr_d,weight_decay=1e-7)
    netd.load_state_dict(torch.load('model parameters/epoch_model_d.pt'))
    netg.load_state_dict(torch.load('model parameters/epoch_model_g.pt'))
    netd.to('cuda')
    netg.to('cuda')
    criterion = BCELoss()
    bar = tqdm(range(epoch))
    for e in bar:
        total_loss_g = []
        total_loss_d = []
        losses_g,losses_d = cross_train(bar,r_data,netg,netd,criterion,optimizer_d,optimizer_g)
        bar.set_description(f'1 epoch Training_G loss:{losses_g}')
        bar.set_description(f'1 epoch Training_D loss:{losses_d}')
        vision_results(netg,netd)
        total_loss_g.append(losses_g)
        total_loss_d.append(losses_d)




    
    