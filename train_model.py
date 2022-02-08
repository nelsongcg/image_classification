#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import boto3
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import io
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from smdebug.pytorch import get_hook
import torch.nn.functional as F
import logging
import sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
from smdebug import modes
import glob
#TODO: Import dependencies for Debugging andd Profiling


class dogBreedsDataset(Dataset):
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        classes_names =  os.listdir(self.data_dir)
        classes_names.sort()
        
        self.image_list = []
        for i,c in enumerate(classes_names):
            label = int(c[:3])
            file_names = os.listdir(os.path.join(data_dir,c))
            file_names = list(filter(lambda k: 'jpg' in k, file_names))
            file_names.sort()
            self.image_list+=zip([c]*len(file_names),file_names,[label]*len(file_names))
        
    
    def __len__(self):
        return len(self.image_list)
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data_dir, self.image_list[idx][0],self.image_list[idx][1])
        label = self.image_list[idx][2]
        image = Image.open(img_name)

        if self.transform:
            sample = self.transform(image)

        return (sample,label)
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()])
        
    
    

def test(model, test_loader,criterion):
    hook = get_hook(create_if_not_exists=True)
    
    if hook:
            hook.set_mode(modes.EVAL)
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1] 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def train(model, train_loader, criterion, optimizer):
    
    hook = get_hook(create_if_not_exists=True)
    
    if hook:
        hook.register_loss(criterion)

    for epoch in range(1, args.epoch+1):
        if hook:
            hook.set_mode(modes.TRAIN)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)            
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    return model
    
def net():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 134)) #WHY USE 134 ??

    return model

def create_data_loaders(data_dir, batch_size):

    s3 = boto3.resource('s3')
    bucket =  s3.Bucket('sagemaker-ap-northeast-1-985768962182')
    s3_folder = 'dogImages'
    local_dir = None
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)



    #create data loader
    dogbreeds = dogBreedsDataset(data_dir=data_dir)
    
    data_loader = DataLoader(dogbreeds, batch_size, shuffle=True)
    
    return data_loader

def main(args):
    
    #Getting a pretrained model (RESNET)
    model=net()
    
    
    # cross entropy will be used as a loss function and opimitizer will be Adam
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    
    #get data loaders
    train_loader = create_data_loaders(data_dir='dogImages/train',batch_size=args.batch_size)
    test_loader = create_data_loaders(data_dir='dogImages/test',batch_size=args.batch_size)

    #train model
    model=train(model, train_loader, loss_criterion, optimizer)
    
    #test model
    test(model, test_loader,criterion=loss_criterion)
    
    #save model
    torch.save(model, 'model.pth')

if __name__=='__main__':
    
    #Defining trainining arguments arguments
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    
    args=parser.parse_args()
    
    main(args)
