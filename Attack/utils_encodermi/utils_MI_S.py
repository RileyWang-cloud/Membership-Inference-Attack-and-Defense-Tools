from datetime import datetime
from functools import partial
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet
from tqdm import tqdm
import argparse
import json
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import h5py

import utils.utils_MI_S_classifier as classifier
import utils.utils_MI_S_modelnet as modelnet
import torch.optim as optim
from torch.autograd import Variable
from tqdm import trange

parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')

parser.add_argument('-a', '--arch', default='resnet18')
# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
parser.add_argument('--moco-k', default=4096, type=int, help='queue size; number of negative keys')
parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')
parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')
parser.add_argument('--bn-splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')
parser.add_argument('--symmetric', action='store_true', help='use a symmetric loss function that backprops to both crops')

# utils
parser.add_argument('--batch_size_attack', default=200, type=int, metavar='N', help='mini-batch size for training attack model')
parser.add_argument('--attack_model', type=str, default='nn')
parser.add_argument('--attack_learning_rate', type=float, default=0.0001)
parser.add_argument('--attack_batch_size', type=int, default=100)
parser.add_argument('--attack_n_hidden', type=int, default=256)
parser.add_argument('--attack_epochs', type=int, default=300)
parser.add_argument('--attack_l2_ratio', type=float, default=1e-6)

parser.add_argument('--pretrain_target_path', type=str, default='./cache-target-encoder/model_last.pth')
parser.add_argument('--pretrain_shadow_path', type=str, default='./cache-shadow-encoder/model_last.pth')

args = parser.parse_args()  # running in command line

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

attack_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale = (0.2, 1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
])

test_attack_transform = transforms.Compose([
    attack_transform,
    test_transform
])

# data prepare
memory_data_attack = CIFAR10(root='data', train=True, transform=test_attack_transform, download=True)
test_data_attack = CIFAR10(root='data', train=False, transform=test_attack_transform, download=True)

"""
Define base encoder
"""
# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)


class ModelBase_output_feature_vector(nn.Module):
    """
    Common CIFAR ResNet recipe. Without linear projection head.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, feature_dim=128, arch=None, bn_splits=16):
        super(ModelBase_output_feature_vector, self).__init__()

        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
                continue
            self.net.append(module)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        return x

"""
load_target_indices 
"""
def load_target_indices():
    fname = './data_indices/target_data_indices.npz'
    with np.load(fname) as f:
        indices = [f['arr_%d' % i] for i in range(len(f.files))]
    return indices

"""
load_shadow_indices
"""
def load_shadow_indices():
    fname = './data_indices/shadow_data_indices.npz'
    with np.load(fname) as f:
        indices = [f['arr_%d' % i] for i in range(len(f.files))]
    return indices

"""
load target/shadow encoder
"""
def load_encoder(model, pretrained_path):
    if os.path.isfile(pretrained_path):
        print("=> loading checkpoint '{}'".format(pretrained_path))
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder_q') and not k.startswith('encoder_q.net.9'):
                # remove prefix
                state_dict[k[len("encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("=> loaded encoder'{}'=<".format(pretrained_path))
        return model
    else:
        print("=> no checkpoint found at '{}'".format(pretrained_path))
        return model


'''
obtain feature vectors' cosine similarity directly
'''
def our_attack_save_train(model):
    # load train/test data of target model
    model = load_encoder(model, args.pretrain_shadow_path)
    
    for i in range(10):
        exec('attack_x_{}, attack_y_{} = [], []'.format(i, i))
    # load target model's traininig & testing data
    indices_shadow = load_shadow_indices()[0]
    indices_shadow_train = indices_shadow [:10000]
    indices_shadow_test = indices_shadow [10000: 20000]

    for i in range(10):
        memory_data_shadow = torch.utils.data.Subset(dataset=memory_data_attack, indices=indices_shadow_train)
        memory_sub_loader = DataLoader(memory_data_shadow, batch_size=args.batch_size_attack, shuffle=False, num_workers=16, pin_memory=True)
        test_data_shadow = torch.utils.data.Subset(dataset= memory_data_attack, indices=indices_shadow_test)
        test_sub_loader = DataLoader(test_data_shadow, batch_size=args.batch_size_attack, shuffle=False, num_workers=16, pin_memory=True)
        memory_bar = tqdm(memory_sub_loader)
        test_bar = tqdm(test_sub_loader)

        print('='*20+'Save training features via RandomCrop-#'+str(i)+'='*20)
        for data, target in memory_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = model(data)
            feature = F.normalize(feature, dim=1)
            feature = feature.cpu().detach().numpy()
            exec('attack_x_{}.append(feature)'.format(i)) 
            # data used in training, label is 1
            exec('attack_y_{}.append(np.ones(args.batch_size_attack))'.format(i))

        print('='*20+'Save testing features via RandomCrop-#'+str(i)+'='*20)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = model(data)
            feature = F.normalize(feature, dim=1)
            feature = feature.cpu().detach().numpy()
            exec('attack_x_{}.append(feature)'.format(i)) 
            # data used in training, label is 1
            exec('attack_y_{}.append(np.zeros(args.batch_size_attack))'.format(i))

        exec('attack_x_{} = np.vstack(attack_x_{})'.format(i,i))
        exec('attack_y_{} = np.concatenate(attack_y_{})'.format(i,i))

    train_data = []
    for k in range(10000):
        features = []
        point = []
        for i in range(10):
            exec('features.append(np.array(attack_x_{}[k]))'.format(i))
        features = np.array(features)
        cos_matrix = cosine_similarity(features)
        for a in range(10):
            for b in range(a+1, 10):
                point.append(np.array([0,0,cos_matrix[a,b]]))
        point = np.vstack(point)
        train_data.append(point)
    train_data = np.array(train_data)
    train_label = np.ones(10000)

    test_data = []
    for k in range(10000,20000):
        features = []
        point = []
        for i in range(10):
            exec('features.append(np.array(attack_x_{}[k]))'.format(i))
        features = np.array(features)
        cos_matrix = cosine_similarity(features)
        for a in range(10):
            for b in range(a+1, 10):
                point.append(np.array([0,0,cos_matrix[a,b]]))
        point = np.vstack(point)
        test_data.append(point)
    test_data = np.array(test_data)
    test_label = np.zeros(10000)

    if not os.path.exists('./dump/'):
        os.mkdir('./dump/')
    f = h5py.File('./dump/MI_set_based_shadow_train_data.h5','w') 
    f['data'] = train_data 
    f['label'] = train_label 
    f.close()  
    f = h5py.File('./dump/MI_set_based_shadow_test_data.h5','w') 
    f['data'] = test_data 
    f['label'] = test_label 
    f.close()


# obtain feature vectors' cosine similarity directly
def our_attack_save_test(model):
    # load train/test data of target model
    model = load_encoder(model, args.pretrain_target_path)
    
    for i in range(10):
        exec('attack_x_{}, attack_y_{} = [], []'.format(i, i))
    # load target model's traininig & testing data
    indices_target = load_target_indices()[0]

    for i in range(10):
        memory_data_target = torch.utils.data.Subset(dataset=memory_data_attack, indices=indices_target)
        memory_sub_loader = DataLoader(memory_data_target, batch_size=args.batch_size_attack, shuffle=False, num_workers=16, pin_memory=True)
        test_sub_loader = DataLoader(test_data_attack, batch_size=args.batch_size_attack, shuffle=False, num_workers=16, pin_memory=True)
        memory_bar = tqdm(memory_sub_loader)
        test_bar = tqdm(test_sub_loader)

        print('='*20+'Save training features via RandomCrop-#'+str(i)+'='*20)
        for data, target in memory_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = model(data)
            feature = F.normalize(feature, dim=1)
            feature = feature.cpu().detach().numpy()
            exec('attack_x_{}.append(feature)'.format(i)) 
            # data used in training, label is 1
            exec('attack_y_{}.append(np.ones(args.batch_size_attack))'.format(i))

        print('='*20+'Save testing features via RandomCrop-#'+str(i)+'='*20)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = model(data)
            feature = F.normalize(feature, dim=1)
            feature = feature.cpu().detach().numpy()
            exec('attack_x_{}.append(feature)'.format(i)) 
            # data used in training, label is 1
            exec('attack_y_{}.append(np.zeros(args.batch_size_attack))'.format(i))

        exec('attack_x_{} = np.vstack(attack_x_{})'.format(i,i))
        exec('attack_y_{} = np.concatenate(attack_y_{})'.format(i,i))

    train_data = []
    for k in range(10000):
        features = []
        point = []
        for i in range(10):
            exec('features.append(np.array(attack_x_{}[k]))'.format(i))
        features = np.array(features)
        cos_matrix = cosine_similarity(features)
        for a in range(10):
            for b in range(a+1, 10):
                point.append(np.array([0,0,cos_matrix[a,b]]))
        point = np.vstack(point)
        train_data.append(point)
    train_data = np.array(train_data)
    train_label = np.ones(10000)

    test_data = []
    for k in range(10000,20000):
        features = []
        point = []
        for i in range(10):
            exec('features.append(np.array(attack_x_{}[k]))'.format(i))
        features = np.array(features)
        cos_matrix = cosine_similarity(features)
        for a in range(10):
            for b in range(a+1, 10):
                point.append(np.array([0,0,cos_matrix[a,b]]))
        point = np.vstack(point)
        test_data.append(point)
    test_data = np.array(test_data)
    test_label = np.zeros(10000)

    if not os.path.exists('./dump/'):
        os.mkdir('./dump/')
    f = h5py.File('./dump/MI_set_based_target_train_data.h5','w') 
    f['data'] = train_data 
    f['label'] = train_label 
    f.close()
    f = h5py.File('./dump/MI_set_based_target_test_data.h5','w') 
    f['data'] = test_data 
    f['label'] = test_label 
    f.close()

'''
Based on: https://github.com/manzilzaheer/DeepSets/blob/master/PointClouds/run.py
'''
class PointCloudTrainer(object):
    def __init__(self, batch_size, downsample, network_dim, num_epochs):
        #Data loader
        self.model_fetcher = modelnet.ModelFetcher(batch_size, downsample, do_standardize=True, do_augmentation=True)

        #Setup network
        self.D = classifier.DTanh(network_dim, pool='max1').cuda()
        self.L = nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.Adam([{'params':self.D.parameters()}], lr=1e-3, weight_decay=1e-7, eps=1e-3)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=list(range(400,num_epochs,400)), gamma=0.1)
        self.num_epochs = num_epochs

    def train(self):
        self.D.train()
        loss_val = float('inf')
        for j in trange(self.num_epochs, desc="Epochs: "):
            counts = 0
            sum_acc = 0.0
            train_data = self.model_fetcher.train_data(loss_val)
            for x, _, y in train_data:
                counts += len(y)
                X = Variable(torch.cuda.FloatTensor(x))
                Y = Variable(torch.cuda.LongTensor(y))
                self.optimizer.zero_grad()
                f_X = self.D(X)
                loss = self.L(f_X, Y)
                # loss_val = loss.data.cpu().numpy()[0]
                loss_val = loss.data.cpu().numpy()
                # sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()[0]
                sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()
                train_data.set_description('Train loss: {0:.4f}'.format(loss_val))
                loss.backward()
                classifier.clip_grad(self.D, 5)
                self.optimizer.step()
                del X,Y,f_X,loss
            train_acc = sum_acc/counts
            self.scheduler.step()

    def test(self):
        self.D.eval()
        counts = 0
        sum_acc = 0.0
        for x, _, y in self.model_fetcher.test_data():
            counts += len(y)
            X = Variable(torch.cuda.FloatTensor(x))
            Y = Variable(torch.cuda.LongTensor(y))
            f_X = self.D(X)
            sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()
            # sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()[0]
            del X,Y,f_X
        test_acc = sum_acc/counts
        print('Final Test Accuracy: {0:0.3f}'.format(test_acc))
        return test_acc

    def pre_recall(self):
        self.D.eval()
        counts = 0
        sum_tp, sum_retrieved = 0.0, 0.0
        for x, _, y in self.model_fetcher.test_data():
            counts += len(y)
            X = Variable(torch.cuda.FloatTensor(x))
            Y = Variable(torch.cuda.LongTensor(y))
            f_X = self.D(X)
            pred = f_X.max(dim=1)[1].float().data.cpu().numpy()
            for i in range(len(y)):
                if pred[i]==1 and y[i]==1:
                    sum_tp += 1
                if pred[i]==1:
                    sum_retrieved +=1
            del X,Y,f_X
        precision = sum_tp / sum_retrieved
        recall = sum_tp / 10000
        print('Final Precision: {0:0.3f}'.format(precision))
        print('Final Recall: {0:0.3f}'.format(recall))

        return precision, recall