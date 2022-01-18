import torch
from  config import *
import numpy as np
from torchvision.utils import make_grid,save_image
import os
import torch.nn as nn
import matplotlib.pyplot as plt


def train_G(e,data,model_g:nn.Module,model_d:nn.Module,criterion,optimizer_g,device:str='cuda'):
    losses = []
    for i in  data:
        noise = torch.randn([batch_size,noise_dim])
        fake_img = model_g(noise.to(device))
        output = model_d(fake_img)
        r_label = torch.ones_like(output)
        loss_g = criterion(output,r_label.to(device))
        losses.append(loss_g.item())
        model_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()
    if e%2==1:
        model_g.eval()
        fake_img = model_g(torch.randn([batch_size,noise_dim]).cuda())
        grid_imgs = make_grid(fake_img.to('cpu'))
        plt.figure(figsize=(10.8,9.6))
        plt.imshow(grid_imgs.permute(1,2,0))
        plt.savefig(os.path.join(os.getcwd(),'vision_picture.png'))
        plt.imshow()
        abs_mdoel_dir = os.path.join(os.getcwd(),model_dir)
        if not os.path.exists(abs_mdoel_dir):
            os.makedirs(abs_mdoel_dir)
        torch.save(model_g.state_dict(),os.path.join(abs_mdoel_dir,f'epoch_model_g.pt'))
        torch.save(model_d.state_dict(),os.path.join(abs_mdoel_dir,f'epoch_model_d.pt'))
    return np.mean(losses)

def train_D(data,model_g,model_d,criterion,optimizer_d,device:str='cuda'):
    losses = []
    for i,imgs in enumerate(data):
        r_imgs = imgs.to(device)
        noise = torch.randn([batch_size,noise_dim])
        f_imgs = model_g(noise.to(device))
        output_r = model_d(r_imgs)
        output_f = model_d(f_imgs.detach())
        r_label = torch.ones_like(output_r)
        f_label = torch.zeros_like(output_f)
        loss_r = criterion(output_r,r_label.to(device))
        loss_f = criterion(output_f,f_label.to(device))
        loss = (loss_f+loss_r)/2
        losses.append(loss.item())
        model_d.zero_grad()
        loss.backward()
        optimizer_d.step()
    return np.mean(losses)

def cross_train(bar, data,model_g,model_d,criterion,optimizer_d,optimizer_g,device:str='cuda'):
    losses_g = []
    losses_d = []
    for i,imgs in enumerate(data):
        '''
        train_d
        '''
        model_d.train()
        r_imgs = imgs.to(device)
        noise = torch.randn([batch_size,noise_dim])
        f_imgs = model_g(noise.to(device))
        output_r = model_d(r_imgs.detach())
        output_f = model_d(f_imgs.detach())
        r_label = torch.ones_like(output_r)
        f_label = torch.zeros_like(output_f)
        loss_r = criterion(output_r,r_label.to(device))
        loss_f = criterion(output_f,f_label.to(device))
        loss = (loss_r+loss_f)/2
        model_d.zero_grad()
        loss.backward()
        losses_d.append(loss.item())
        optimizer_d.step()
        bar.set_description(f'1 batch loss_d:{loss.item()}')
        '''
        train_g
        '''
        model_g.train()
        noise = torch.randn([batch_size,noise_dim])
        fake_img = model_g(noise.to(device))
        output = model_d(fake_img)
        r_label = torch.ones_like(output)
        loss_g = criterion(output,r_label.to(device))
        losses_g.append(loss_g.item())
        model_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()
        bar.set_description(f'1 batch loss_g:{loss_g.item()}')
    return np.mean(losses_d),np.mean(losses_g)
    
def vision_results(model_g,model_d):
        model_g.eval()
        fake_img = model_g(torch.randn([batch_size,noise_dim]).cuda())
        fake_img = (fake_img+1.0)/2
        grid_imgs = make_grid(fake_img.to('cpu'))
        save_image(grid_imgs,os.path.join(os.getcwd(),'vision_picture_2.png'))
        abs_mdoel_dir = os.path.join(os.getcwd(),model_dir)
        if not os.path.exists(abs_mdoel_dir):
            os.makedirs(abs_mdoel_dir)
        torch.save(model_g.state_dict(),os.path.join(abs_mdoel_dir,f'epoch_model_g.pt'))
        torch.save(model_d.state_dict(),os.path.join(abs_mdoel_dir,f'epoch_model_d.pt'))
        model_g.train()
