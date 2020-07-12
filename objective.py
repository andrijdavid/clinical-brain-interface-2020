import numpy as np
from hyperopt import  STATUS_OK
from timeseries import *
from models import *
import pickle 
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from models.layers import *
import neptune
from neptunecontrib.monitoring.fastai import NeptuneMonitor
from pathlib import Path
import gc
import names
from fastai.utils.mod_display import *

class SeModule(nn.Module):
    def __init__(self, ch, reduction, act_fn='relu'):
        super().__init__()
        nf = math.ceil(ch//reduction/8)*8
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = convlayer(ch, nf, 1, act_fn=act_fn)
        self.conv2 = convlayer(nf, ch, 1, act_fn=False)
        
    def forward(self, x):
        res = self.pool(x)
        res = self.conv1(res)
        res = self.conv2(res)
        res = nn.functional.sigmoid(res)
        return x * res
 
       
class ResBlock(nn.Module):
    def __init__(self, ni, nf, ks=[7, 5, 3], act_fn='relu'):
        super().__init__()
        self.conv1 = convlayer(ni, nf, ks[0], act_fn=act_fn)
        self.conv2 = convlayer(nf, nf, ks[1], act_fn=act_fn)
        self.conv3 = convlayer(nf, nf, ks[2], act_fn=False, zero_bn=True)
        self.se = SeModule(nf, 16, act_fn=act_fn)
        # expand channels for the sum if necessary
        self.shortcut = noop if ni == nf else convlayer(ni, nf, ks=1, act_fn=False)
        self.act_fn = get_act_layer(act_fn)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Squeeze and Excitation
        x = self.se(x)
        # Shortcut
        sc = self.shortcut(res)
        # Residual
        x += sc
        x = self.act_fn(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self,c_in, c_out, nf = 64, pool=nn.AdaptiveAvgPool1d, act_fn='relu'):
        super().__init__()
        self.block1 = ResBlock(c_in, nf, ks=[7, 5, 3], act_fn=act_fn)
        self.block2 = ResBlock(nf, nf * 2, ks=[7, 5, 3], act_fn=act_fn)
        self.block3 = ResBlock(nf * 2, nf * 2, ks=[7, 5, 3], act_fn=act_fn)
        self.gap = pool(1)
        self.fc = nn.Linear(nf * 2, c_out)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)
    
class FCN(nn.Module):
    def __init__(self,c_in,c_out,layers=[128,256,128],kss=[7,5,3], dilations=[], act_fn='mish'):
        super().__init__()
        self.conv1 = convlayer(c_in,layers[0],kss[0], act_fn=act_fn)
        self.conv2 = convlayer(layers[0],layers[1],kss[1], act_fn=act_fn, dilation=dilations[0])
        self.conv3 = convlayer(layers[1],layers[2],kss[2], act_fn=act_fn, zero_bn=True, dilation=dilations[1])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(layers[-1],c_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)  
    
class ResxBlock(nn.Module):
    def __init__(self, ni, nf, cardinality=12, stride=1, ks=[1,3,1], act_fn='relu'):
        super().__init__()
        self.conv1 = convlayer(ni, nf, ks[0], act_fn=act_fn, bias=False)
        self.conv2 = convlayer(nf, nf, ks[1], act_fn=act_fn, stride=stride, bias=False, padding=1, groups=cardinality)
        self.conv3 = convlayer(nf, nf, ks[2], act_fn=False, bias=False)
        self.se = SeModule(nf, 16, act_fn=act_fn)
        # expand channels for the sum if necessary
        self.shortcut = noop if ni == nf else convlayer(ni, nf, ks=1, act_fn=False, stride=stride, bias=False)
        self.act_fn = get_act_layer(act_fn)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.se(x)
        sc = self.shortcut(res)
        x += sc
        x = self.act_fn(x)
        return x
    
class ResNeXt(nn.Module):
    def __init__(self,c_in, c_out, cardinality=2, bottleneck_width=64, pool=nn.AdaptiveAvgPool1d, act_fn='mish'):
        super().__init__()
        nf = cardinality * bottleneck_width
        self.block1 = ResxBlock(c_in, nf, act_fn=act_fn, stride=1, cardinality=cardinality)
        self.block2 = ResxBlock(nf, nf * 2, act_fn=act_fn, stride=2, cardinality=cardinality)
        self.block3 = ResxBlock(nf*2, nf*2*2, act_fn=act_fn, stride=2, cardinality=cardinality)
        self.gap = pool(1)
        self.fc = nn.Linear(nf*2*2, c_out)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)
    
x, y = pickle.load(open(Path("./data/train.pkl"), "rb"))
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42123456)

def objfcn(args):
    d1, d2, act_fn, scale_by_channel,scale_by_sample, scale_type, randaugment = args    
    scale_range = (-1, 1)
    bs = 128
    data = (ItemLists(Path("data"), TSList(x_train),TSList(x_val))
        .label_from_lists(y_train, y_val)
        .databunch(bs=bs, val_bs=bs * 2)
        .scale(scale_type=scale_type, scale_by_channel=scale_by_channel, 
             scale_by_sample=scale_by_sample,scale_range=scale_range)
    )
    model = FCN(data.features, data.c, act_fn=act_fn, dilations=[d1, d2])
    neptune.init(project_qualified_name='andrijdavid/ClinicalBrainComputerInterfacesChallenge2020')
    neptune.create_experiment(name=f'FCN Hyperparamter Search', description="Optimizing accuracy by searching proper dilation", params={
        'pool': 'AdaptiveAvgPool1d',
        'dilation1': d1,
        'dilation2': d2,
        'act_fn': act_fn,
        'scale_by_channel': scale_by_channel,
        'scale_by_sample': scale_by_sample,
        'scale_type': scale_type,
        'randaugment': randaugment,
        'bs': bs,
        'model': 'fcn',
        'epoch': 100
    }, tags =['hyperopt'])
    kappa = KappaScore()
    learn = Learner(data, model, metrics=[accuracy, kappa])
    if randaugment:
        learn = learn.randaugment()
    learn.fit_one_cycle(100, callbacks=[NeptuneMonitor()])
    val = learn.validate()
    learn.destroy()
    data = None
    neptune.stop()
    return {
        'loss': 1 - (val[1].item()),
        'status': STATUS_OK,
        'kappa': val[-1].item()
    }

def obj(args):
    nf, pool, act_fn, scale_by_channel,scale_by_sample, scale_type, randaugment = args    
    scale_range = (-1, 1)
    bs = 32
    data = (ItemLists(Path("data"), TSList(x_train),TSList(x_val))
        .label_from_lists(y_train, y_val)
        .databunch(bs=bs, val_bs=bs * 2)
        .scale(scale_type=scale_type, scale_by_channel=scale_by_channel, 
             scale_by_sample=scale_by_sample,scale_range=scale_range)
    )
    model = ResNet(data.features, data.c, act_fn=act_fn, nf=nf, pool=pool)
    neptune.init(project_qualified_name='andrijdavid/ClinicalBrainComputerInterfacesChallenge2020')
    neptune.create_experiment(name=f'ResNet Hyperparamter Search', description="Optimizing accuracy", params={
        'nf': nf,
        'pool': pool,
        'act_fn': act_fn,
        'scale_by_channel': scale_by_channel,
        'scale_by_sample': scale_by_sample,
        'scale_type': scale_type,
        'randaugment': randaugment,
        'bs': bs,
        'model': 'resnet',
        'epoch': 100
    }, tags =['hyperopt'])
    kappa = KappaScore()
    learn = Learner(data, model, metrics=[accuracy, kappa])
    if randaugment:
        learn = learn.randaugment()
    learn.fit_one_cycle(100, callbacks=[NeptuneMonitor()])
    val = learn.validate()
    learn.destroy()
    data = None
    neptune.stop()
    return {
        'loss': 1 - (val[1].item()),
        'status': STATUS_OK,
        'kappa': val[-1].item()
    }

def obj2(args):
    nf, act_fn, scale_by_channel,scale_by_sample, scale_type = args    
    scale_range = (-1, 1)
    bs = 32
    data = (ItemLists(Path("data"), TSList(x_train),TSList(x_val))
        .label_from_lists(y_train, y_val)
        .databunch(bs=bs, val_bs=bs * 2)
        .scale(scale_type=scale_type, scale_by_channel=scale_by_channel, 
             scale_by_sample=scale_by_sample,scale_range=scale_range)
    )
    model = ResNet(data.features, data.c, act_fn=act_fn, nf=nf)
    neptune.init(project_qualified_name='andrijdavid/ClinicalBrainComputerInterfacesChallenge2020')
    neptune.create_experiment(name=f'ResNet Hyperparamter Search', description="Optimizing accuracy", params={
        'nf': nf,
        'act_fn': act_fn,
        'scale_by_channel': scale_by_channel,
        'scale_by_sample': scale_by_sample,
        'scale_type': scale_type,
        'bs': bs,
        'model': 'resnet',
        'epoch': 100
    }, tags =['hyperopt'])
    name = names.get_first_name()
#     kappa = KappaScore()
    loss_func = LabelSmoothingCrossEntropy()
    learn = Learner(data, model, metrics=[accuracy], loss_func=loss_func, opt_func=Ranger)
    with progress_disabled_ctx(learn) as learn:
        learn.fit_one_cycle(100, callbacks=[NeptuneMonitor()])
    learn.save(f"{name}")
    val = learn.validate()
    learn.destroy()
    data = None
    neptune.log_artifact(f'data/models/{name}.pth')
    neptune.stop()
    return {
        'loss': 1 - (val[1].item()),
        'status': STATUS_OK,
        'kappa': val[-1].item()
    }

def objx(args):
    c, act_fn, scale_by_channel,scale_by_sample, scale_type = args    
    scale_range = (-1, 1)
    bs = 16
    data = (ItemLists(Path("data"), TSList(x_train),TSList(x_val))
        .label_from_lists(y_train, y_val)
        .databunch(bs=bs, val_bs=bs * 2)
        .scale(scale_type=scale_type, scale_by_channel=scale_by_channel, 
             scale_by_sample=scale_by_sample,scale_range=scale_range)
    )
    model = ResNext(data.features, data.c, cardinality=c, act_fn=act_fn)
    neptune.init(project_qualified_name='andrijdavid/ClinicalBrainComputerInterfacesChallenge2020')
    neptune.create_experiment(name=f'ResNet Hyperparamter Search', description="Optimizing accuracy", params={
        'cardinality': c,
        'act_fn': act_fn,
        'scale_by_channel': scale_by_channel,
        'scale_by_sample': scale_by_sample,
        'scale_type': scale_type,
        'bs': bs,
        'model': 'resnext',
        'epoch': 100
    }, tags =['hyperopt'])
    name = names.get_first_name()
    kappa = KappaScore()
    learn = Learner(data, model, metrics=[accuracy, kappa])
    learn.fit_one_cycle(100, callbacks=[NeptuneMonitor()])
    learn.save(f"{name}")
    val = learn.validate()
    learn.destroy()
    data = None
    neptune.log_artifact(f'data/models/{name}.pth')
    neptune.stop()
    return {
        'loss': 1 - (val[1].item()),
        'status': STATUS_OK,
        'kappa': val[-1].item()
    }