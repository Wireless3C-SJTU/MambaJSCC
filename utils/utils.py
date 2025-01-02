# --------------------------------------------------------
# Modified By Mzero
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
from math import inf
import torch
import torch.distributed as dist
from timm.utils import ModelEma as ModelEma
import numpy as np
import random
from torch.optim.lr_scheduler import _LRScheduler

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

def save_model(model, save_path):
    torch.save(model, save_path)


import os
import torch
def check_gpus():
    '''
    GPU available check
    http://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-cuda/
    '''
    if not torch.cuda.is_available():
        print('This script could only be used to manage NVIDIA GPUs,but no GPU found in your device')
        return False
    elif not 'NVIDIA System Management' in os.popen('nvidia-smi -h').read():
        print("'nvidia-smi' tool not found.")
        return False
    return True


def parse(line,qargs):
    '''
    line:
        a line of text
    qargs:
        query arguments
    return:
        a dict of gpu infos
    Pasing a line of csv format text returned by nvidia-smi
    解析一行nvidia-smi返回的csv格式文本
    '''
    numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']#可计数的参数
    power_manage_enable=lambda v:(not 'Not Support' in v)#lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
    to_numberic=lambda v:float(v.upper().strip().replace('MIB','').replace('W',''))#带单位字符串去掉单位
    process = lambda k,v:((int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
    return {k:process(k,v) for k,v in zip(qargs,line.strip().split(','))}

def query_gpu(qargs=[]):
    '''
    qargs:
        query arguments
    return:
        a list of dict
    Querying GPUs infos
    查询GPU信息
    '''
    qargs =['index','gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']+ qargs
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()
    return [parse(line,qargs) for line in results]

def by_power(d):
    '''
    helper function fo sorting gpus by power
    '''
    power_infos=(d['power.draw'],d['power.limit'])
    if any(v==1 for v in power_infos):
        print('Power management unable for GPU {}'.format(d['index']))
        return 1
    return float(d['power.draw'])/d['power.limit']

class GPUManager():
    '''
    qargs:
        query arguments
    A manager which can list all available GPU devices
    and sort them and choice the most free one.Unspecified 
    ones pref.
    GPU设备管理器，考虑列举出所有可用GPU设备，并加以排序，自动选出
    最空闲的设备。在一个GPUManager对象内会记录每个GPU是否已被指定，
    优先选择未指定的GPU。
    '''
    def __init__(self,qargs=[]):
        '''
        '''
        self.qargs=qargs
        self.gpus=query_gpu(qargs)
        for gpu in self.gpus:
            gpu['specified']=False
        self.gpu_num=len(self.gpus)

    def _sort_by_memory(self,gpus,by_size=False):
        if by_size:
            print('Sorted by free memory size')
            return sorted(gpus,key=lambda d:d['memory.free'],reverse=True)
        else:
            print('Sorted by free memory rate')
            return sorted(gpus,key=lambda d:float(d['memory.free'])/ d['memory.total'],reverse=True)

    def _sort_by_power(self,gpus):
        return sorted(gpus,key=by_power)
    
    def _sort_by_custom(self,gpus,key,reverse=False,qargs=[]):
        if isinstance(key,str) and (key in qargs):
            return sorted(gpus,key=lambda d:d[key],reverse=reverse)
        if isinstance(key,type(lambda a:a)):
            return sorted(gpus,key=key,reverse=reverse)
        raise ValueError("The argument 'key' must be a function or a key in query args,please read the documention of nvidia-smi")

    def auto_choice(self,mode=3):
        '''
        mode:
            0:(default)sorted by free memory size
        return:
            a TF device object
        Auto choice the freest GPU device,not specified
        ones 
        自动选择最空闲GPU,返回索引
        '''
        for old_infos,new_infos in zip(self.gpus,query_gpu(self.qargs)):
            old_infos.update(new_infos)
        unspecified_gpus=[gpu for gpu in self.gpus if not gpu['specified']] or self.gpus
        
        if mode==0:
            #print('Choosing the GPU device has largest free memory...')
            chosen_gpu=self._sort_by_memory(unspecified_gpus,True)
        elif mode==1:
            #print('Choosing the GPU device has highest free memory rate...')
            chosen_gpu=self._sort_by_power(unspecified_gpus)
        elif mode==2:
            #print('Choosing the GPU device by power...')
            chosen_gpu=self._sort_by_power(unspecified_gpus)
        else:
            #print('Given an unaviliable mode,will be chosen by memory')
            chosen_gpu=self._sort_by_memory(unspecified_gpus)
        if int(chosen_gpu[0]['index'])==3:
            chosen_gpu=chosen_gpu[1]
        else:
            chosen_gpu=chosen_gpu[0]
        chosen_gpu['specified']=True
        index=chosen_gpu['index']
        #print('Using GPU {i}:\n{info}'.format(i=index,info='\n'.join([str(k)+':'+str(v) for k,v in chosen_gpu.items()])))
        return int(index)


# def load_checkpoint_ema(config, model, optimizer, lr_scheduler, loss_scaler, logger, model_ema: ModelEma=None):
#     logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
#     if config.MODEL.RESUME.startswith('https'):
#         checkpoint = torch.hub.load_state_dict_from_url(
#             config.MODEL.RESUME, map_location='cpu', check_hash=True)
#     else:
#         checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    
#     if 'model' in checkpoint:
#         msg = model.load_state_dict(checkpoint['model'], strict=False)
#         logger.info(f"resuming model: {msg}")
#     else:
#         logger.warning(f"No 'model' found in {config.MODEL.RESUME}! ")

#     if model_ema is not None:
#         if 'model_ema' in checkpoint:
#             msg = model_ema.ema.load_state_dict(checkpoint['model_ema'], strict=False)
#             logger.info(f"resuming model_ema: {msg}")
#         else:
#             logger.warning(f"No 'model_ema' found in {config.MODEL.RESUME}! ")

#     max_accuracy = 0.0
#     max_accuracy_ema = 0.0
#     if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#         config.defrost()
#         config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
#         config.freeze()
#         if 'scaler' in checkpoint:
#             loss_scaler.load_state_dict(checkpoint['scaler'])
#         logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
#         if 'max_accuracy' in checkpoint:
#             max_accuracy = checkpoint['max_accuracy']
#         if 'max_accuracy_ema' in checkpoint:
#             max_accuracy_ema = checkpoint['max_accuracy_ema']

#     del checkpoint
#     torch.cuda.empty_cache()
#     return max_accuracy, max_accuracy_ema


# def load_pretrained_ema(config, model, logger, model_ema: ModelEma=None, load_ema_separately=False):
#     logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
#     checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    
#     if 'model' in checkpoint:
#         msg = model.load_state_dict(checkpoint['model'], strict=False)
#         logger.warning(msg)
#         logger.info(f"=> loaded 'model' successfully from '{config.MODEL.PRETRAINED}'")
#     else:
#         logger.warning(f"No 'model' found in {config.MODEL.PRETRAINED}! ")

#     if model_ema is not None:
#         key = "model_ema" if load_ema_separately else "model"
#         if key in checkpoint:
#             msg = model_ema.ema.load_state_dict(checkpoint[key], strict=False)
#             logger.warning(msg)
#             logger.info(f"=> loaded '{key}' successfully from '{config.MODEL.PRETRAINED}' for model_ema")
#         else:
#             logger.warning(f"No '{key}' found in {config.MODEL.PRETRAINED}! ")

#     del checkpoint
#     torch.cuda.empty_cache()


# def save_checkpoint_ema(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema: ModelEma=None, max_accuracy_ema=None):
#     save_state = {'model': model.state_dict(),
#                   'optimizer': optimizer.state_dict(),
#                   'lr_scheduler': lr_scheduler.state_dict(),
#                   'max_accuracy': max_accuracy,
#                   'scaler': loss_scaler.state_dict(),
#                   'epoch': epoch,
#                   'config': config}
    
#     if model_ema is not None:
#         save_state.update({'model_ema': model_ema.ema.state_dict(),
#             'max_accuray_ema': max_accuracy_ema})

#     save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
#     logger.info(f"{save_path} saving......")
#     torch.save(save_state, save_path)
#     logger.info(f"{save_path} saved !!!")


# def get_grad_norm(parameters, norm_type=2):
#     if isinstance(parameters, torch.Tensor):
#         parameters = [parameters]
#     parameters = list(filter(lambda p: p.grad is not None, parameters))
#     norm_type = float(norm_type)
#     total_norm = 0
#     for p in parameters:
#         param_norm = p.grad.data.norm(norm_type)
#         total_norm += param_norm.item() ** norm_type
#     total_norm = total_norm ** (1. / norm_type)
#     return total_norm


# def auto_resume_helper(output_dir):
#     checkpoints = os.listdir(output_dir)
#     checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
#     print(f"All checkpoints founded in {output_dir}: {checkpoints}")
#     if len(checkpoints) > 0:
#         latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
#         print(f"The latest checkpoint founded: {latest_checkpoint}")
#         resume_file = latest_checkpoint
#     else:
#         resume_file = None
#     return resume_file


# def reduce_tensor(tensor):
#     rt = tensor.clone()
#     dist.all_reduce(rt, op=dist.ReduceOp.SUM)
#     rt /= dist.get_world_size()
#     return rt


# def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
#     if isinstance(parameters, torch.Tensor):
#         parameters = [parameters]
#     parameters = [p for p in parameters if p.grad is not None]
#     norm_type = float(norm_type)
#     if len(parameters) == 0:
#         return torch.tensor(0.)
#     device = parameters[0].grad.device
#     if norm_type == inf:
#         total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
#     else:
#         total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
#                                                         norm_type).to(device) for p in parameters]), norm_type)
#     return total_norm


# class NativeScalerWithGradNormCount:
#     state_dict_key = "amp_scaler"

#     def __init__(self):
#         self._scaler = torch.cuda.amp.GradScaler()

#     def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
#         self._scaler.scale(loss).backward(create_graph=create_graph)
#         if update_grad:
#             if clip_grad is not None:
#                 assert parameters is not None
#                 self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
#                 norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
#             else:
#                 self._scaler.unscale_(optimizer)
#                 norm = ampscaler_get_grad_norm(parameters)
#             self._scaler.step(optimizer)
#             self._scaler.update()
#         else:
#             norm = None
#         return norm

#     def state_dict(self):
#         return self._scaler.state_dict()

#     def load_state_dict(self, state_dict):
#         self._scaler.load_state_dict(state_dict)

def seed_torch(seed=1024):
    print("seed: ", seed)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False # is False in DIV2K
    torch.backends.cudnn.deterministic = True