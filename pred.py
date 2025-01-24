import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["WORLD_SIZE"] = "1"
#os.environ['LOCAL_RANK'] = '0'
#os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])
#local_rank = int(os.environ["LOCAL_RANK"])
#print("rank, local_rank, world_size:",rank, local_rank, world_size)

from dataset import Dataset_PSDM_train, Dataset_PSDM_val
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.utils.data as Data
import shutil
from guided_diffusion.unet import UNetModel_MS_Former
from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
import torch.distributed as dist
from guided_diffusion.resample import create_named_schedule_sampler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import MultiStepLR
import sklearn
import re
import glob
import os

def list_sort_nicely(l):
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    
    def alphanum_key(s):
        return [ tryint(c) for c in re.split('([0-9]+)', s)]
    l.sort(key=alphanum_key)
    return l

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="0", help='which gpu is used')
parser.add_argument('--bs', type=int, default=2, help='batch size')
parser.add_argument('--T', type=int, default=1000, help='T')
parser.add_argument('--epoch', type=int, default=2000, help='all_epochs')
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--local-rank", default=-1, type=int)
parser.add_argument('--model_path', default="0", type=str)
parser.add_argument('--save_path', default="0", type=str)
parser.add_argument('--ddim', type=str, default='8', help='ddim')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

def ddp_setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '29500'
  dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
  print("setup done")

ddp_setup(rank, world_size)

rank = dist.get_rank()
torch.cuda.set_device(rank)
device = torch.cuda.current_device()
world_size = dist.get_world_size()
print(rank, device, world_size)

device_id = rank % torch.cuda.device_count()
device = torch.device(device_id)

num_gpus = torch.cuda.device_count()

print("Number of available GPUs:", num_gpus)
print("torch.cuda.is_available():", torch.cuda.is_available())

train_bs = args.bs
val_bs = args.bs
lr_max = 0.0001
img_size = (128, 128)
all_epochs = args.epoch

data_root_train = '/home/bolanp1/lee03851/others/DoseDiff/preprocessed_data/NPY/train'
data_root_val = '/home/bolanp1/lee03851/others/DoseDiff/preprocessed_data/NPY/validation'

L2 = 0.0001
ture_bs = len(args.gpu.split(',')) * args.bs
val_bs = 1 * args.bs

save_name = 'T{}_bs{}_epoch{}'.format(args.T, ture_bs, args.epoch)

train_data = Dataset_PSDM_train(data_root=data_root_train)
#train_samper = Data.distributed.DistributedSampler(train_data)
val_data = Dataset_PSDM_val(data_root=data_root_val)
val_samper = Data.distributed.DistributedSampler(val_data)
train_dataloader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=False, num_workers=0, pin_memory=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=val_bs, shuffle=False, num_workers=4, pin_memory=True)

if dist.get_rank() == 0:
    print('train_lenth: %i   val_lenth: %i' % (train_data.len, val_data.len))

dis_channels = 1

model = UNetModel_MS_Former(image_size=img_size, in_channels=1, ct_channels=1, dis_channels=dis_channels,
                       model_channels=128, out_channels=1, num_res_blocks=2, attention_resolutions=(16, 32),
                       dropout=0,
                       channel_mult=(1, 1, 2, 3, 4), conv_resample=True, dims=2, num_classes=None,
                       use_checkpoint=False,
                       use_fp16=False, num_heads=4, num_head_channels=-1, num_heads_upsample=-1,
                       use_scale_shift_norm=True,
                       resblock_updown=False, use_new_attention_order=False)

diffusion = SpacedDiffusion(use_timesteps=space_timesteps(args.T, 'ddim{}'.format(args.ddim)),
                            betas=gd.get_named_beta_schedule("cosine", args.T),
                            model_mean_type=(gd.ModelMeanType.EPSILON),
                            model_var_type=(gd.ModelVarType.FIXED_LARGE),
                            loss_type=gd.LossType.MSE, rescale_timesteps=False)
                            
diffusion_test = SpacedDiffusion(use_timesteps=space_timesteps(args.T, 'ddim4'),
                                betas=gd.get_named_beta_schedule("cosine", args.T),
                                model_mean_type=(gd.ModelMeanType.EPSILON),
                                model_var_type=(gd.ModelVarType.FIXED_LARGE),
                                loss_type=gd.LossType.MSE, rescale_timesteps=False)

device = torch.device(f"cuda:{device_id}")
print('device id:', device_id)
model.cuda()
model = DDP(model.to(device_id), device_ids=[device_id], output_device=device_id, broadcast_buffers=False,
            find_unused_parameters=False)


schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
optimizer = optim.AdamW(model.parameters(), lr=lr_max, weight_decay=L2)
lr_scheduler = MultiStepLR(optimizer, milestones=[int((7 / 10) * args.epoch)], gamma=0.1, last_epoch=-1)
best_MAE = 1000

print("Test code starts")

from PIL import Image
import GPUtil
import nibabel as nib
GPUtil.showUtilization()

model.cuda()
#checkpoint = torch.load(args.model_path)
#model.load_state_dict(checkpoint)
#model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})

checkpoint = torch.load(args.model_path, map_location = f'cuda:{device_id}')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
val_epoch_MAE = []
image_CT = []
ture_rtdose = []
pred_rtdose = []

data_list = list_sort_nicely(glob.glob("/panfs/jay/groups/1/bolanp1/lee03851/others/DoseDiff/preprocessed_data/NPY/validation/ct_dose/*.npy"))[1373:]
print(len(data_list))
j=1
case_list = []

with torch.no_grad():
    for i, (ct_dose, mr_dose, mr) in enumerate(val_dataloader):
        
        case_num = int(data_list[i].split('/')[-1].split('_')[1])
        slice_num =int(data_list[i].split('/')[-1].split('_')[2].split('.')[0])
        ct_dose, mr_dose, mr= ct_dose.cuda().float(), mr_dose.cuda().float(), mr.cuda().float()
        
        ct_dose = ct_dose/35 -1
        mr_dose = mr_dose/35 -1
        
        print(torch.min(ct_dose), torch.max(ct_dose), torch.min(mr_dose), torch.max(mr_dose))
        
        pred = diffusion.ddim_sample_loop(
            model=model, shape=(ct_dose.size(0), 1, img_size[0], img_size[1]), noise=None, clip_denoised=True,
            model_kwargs={'ct': mr_dose, 'dis': mr}, device=None, progress=True, eta=0.0)
        
        pred = (pred+1)*35
        ct_dose = (ct_dose+1)*35
        mr_dose = (mr[:,6,:,:]+1)*35
        head_mask = ct_dose > 0
        
        
        
        pred = pred.detach().cpu().numpy().squeeze()
        ct_dose = ct_dose.detach().cpu().numpy().squeeze()
        mr_dose = mr_dose.detach().cpu().numpy().squeeze()
        head_mask = head_mask.detach().cpu().numpy().squeeze().astype(int)
        pred = pred*head_mask
        pred = np.clip(pred, 0, np.max(mr_dose))
        
        print("pred:", np.min(pred), np.max(pred))
        print("ct_dose:", np.min(ct_dose), np.max(ct_dose))
        print("mr_dose:", np.min(mr_dose), np.max(mr_dose))
        
        #os.makedirs(f"{args.save_path}/mr_dose/", exist_ok = True)
        #os.makedirs(f"{args.save_path}/gt/", exist_ok = True)
        os.makedirs(f"{args.save_path}/pred/", exist_ok = True)
        '''
        mr_dose_nifti = nib.Nifti1Image(mr_dose, affine=np.eye(4))  # Assuming identity affine for simplicity
        nib.save(mr_dose_nifti, f"{args.save_path}/mr_dose/{case_num:03d}_TMR10_dose_{slice_num:03d}.nii.gz")
        
        ct_dose_nifti = nib.Nifti1Image(ct_dose, affine=np.eye(4))  # Assuming identity affine for simplicity
        nib.save(ct_dose_nifti, f"{args.save_path}/gt/{case_num:03d}_Conv_dose_{slice_num:03d}.nii.gz")
        '''
        pred_nifti = nib.Nifti1Image(pred, affine=np.eye(4))  # Assuming identity affine for simplicity
        nib.save(pred_nifti, f"{args.save_path}/pred/{case_num:03d}_sConv_dose_{slice_num:03d}.nii.gz")
        
        
        
        