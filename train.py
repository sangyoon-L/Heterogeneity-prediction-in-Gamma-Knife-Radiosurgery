import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]= '0,1,2,3,4,5,6,7'
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])

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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import MultiStepLR
import re
import glob

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
parser.add_argument("--date", default=-1, type=int)
parser.add_argument("--experiment", default=-1, type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


def ddp_setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '9999'
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

save_name = '{}_T{}_bs{}_epoch{}_{}'.format(args.date, args.T, ture_bs, args.epoch, args.experiment)

if dist.get_rank() == 0:
#    if os.path.exists(os.path.join('trained_models', save_name)):
#        shutil.rmtree(os.path.join('trained_models', save_name))
    os.makedirs(os.path.join('trained_models', save_name), exist_ok=True)
    print(save_name)

    train_writer = SummaryWriter(os.path.join('trained_models', save_name, 'log/train'), flush_secs=2)


train_data = Dataset_PSDM_train(data_root=data_root_train)
train_samper = Data.distributed.DistributedSampler(train_data)
val_data = Dataset_PSDM_val(data_root=data_root_val)
val_samper = Data.distributed.DistributedSampler(val_data)
train_dataloader = DataLoader(dataset=train_data, batch_size=train_bs, sampler=train_samper, shuffle=False, num_workers=0, pin_memory=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=val_bs, sampler=val_samper, shuffle=False, num_workers=0, pin_memory=True)

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

print("args.T:",args.T)

diffusion = SpacedDiffusion(use_timesteps=space_timesteps(args.T, [args.T]),
#diffusion = SpacedDiffusion(use_timesteps=space_timesteps(args.T, 'ddim4'),
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

#model = model.to(f"cuda:{device_id}")
model.cuda()
print('device id:', device_id)
model = DDP(model, device_ids = [device_id], output_device = device_id, broadcast_buffers=False,
           find_unused_parameters=False)
#model = DataParallel(model, device_ids = [0,1])        

#model.load_state_dict(torch.load("/panfs/jay/groups/1/bolanp1/lee03851/others/DoseDiff/trained_models/T1000_bs2_epoch1600_0422/model_epoch6.pth", map_location=f'cuda:{device_id}'))


import GPUtil
import time
GPUtil.showUtilization()
schedule_sampler = create_named_schedule_sampler("uniform", diffusion)


data_list = list_sort_nicely(glob.glob("/panfs/jay/groups/1/bolanp1/lee03851/others/DoseDiff/preprocessed_data/NPY/validation/ct_dose/*.npy"))

print("Train code starts")

checkpoint = torch.load("/home/bolanp1/lee03851/others/NovelDiff/trained_models/241207_T1000_bs128_epoch1600_skull_2_5D/model_epoch0126.pth",map_location=f'cuda:{device_id}')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = optim.AdamW(model.parameters(), lr=lr_max, weight_decay=L2)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

lr_scheduler = MultiStepLR(optimizer, milestones=[int((7 / 10) * args.epoch)], gamma=0.1, last_epoch=126)
best_MAE = 1000

GPUtil.showUtilization()
#for epoch in range(all_epochs):
for epoch in np.arange(126,all_epochs,1):
    lr = optimizer.param_groups[0]['lr']
    print("epoch", epoch + 1)
    model.train()
    train_epoch_loss = []

    for i, (ct_dose, mr_dose, mask) in enumerate(train_dataloader):
        ct_dose, mr_dose, mask = ct_dose.to(device).float(), mr_dose.to(device).float(), mask.to(device).float()

        # Normalize doses
        ct_dose = ct_dose / 35 - 1
        mr_dose = mr_dose / 35 - 1

        optimizer.zero_grad()
        step_time = time.time()
        t, weights = schedule_sampler.sample(ct_dose.shape[0], ct_dose.device)
        
        losses = diffusion.training_losses(model=model, x_start=ct_dose, t=t, model_kwargs={'ct': mr_dose, 'dis': mask}, noise=None)
        loss = (losses["loss"] * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        dist.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
        loss = loss / dist.get_world_size()

        train_epoch_loss.append(loss.item())

        if dist.get_rank() == 0 and i % 10 == 0:
            print('[%d/%d, %d/%d] train_loss: %.3f' %
                  (epoch + 1, all_epochs, i + 1, len(train_dataloader), loss.item()))

    lr_scheduler.step()

    if dist.get_rank() == 0:
        train_epoch_loss = np.mean(train_epoch_loss)
        train_writer.add_scalar('lr', lr, epoch + 1)
        train_writer.add_scalar('train_loss', train_epoch_loss, epoch + 1)

    if (epoch == 0) or (((epoch + 5) % 5) == 0):
        save_epoch = f'{epoch + 1:04d}'
        model.eval()
        val_epoch_MAE = []
        image_CT = []
        ture_rtdose = []
        pred_rtdose = []
        with torch.no_grad():
            for i, (ct_dose, mr_dose, dis) in enumerate(val_dataloader):
                case_num = int(data_list[i].split('/')[-1].split('_')[1])
                slice_num = int(data_list[i].split('/')[-1].split('_')[2].split('.')[0])
                ct_dose, mr_dose, dis = ct_dose.cuda().float(), mr_dose.cuda().float(), dis.cuda().float()

                # Normalize doses
                ct_dose = ct_dose / 35 - 1
                mr_dose = mr_dose / 35 - 1

                pred = diffusion_test.ddim_sample_loop(
                    model=model, shape=(ct_dose.size(0), 1, img_size[0], img_size[1]), noise=None, clip_denoised=True,
                    denoised_fn=None, cond_fn=None, model_kwargs={'ct': mr_dose, 'dis': dis}, device=None, progress=False, eta=0.0)

                pred = (pred + 1) * 35
                ct_dose = (ct_dose + 1) * 35

                head_mask = (ct_dose > 0).float()
                pred = pred * head_mask

                MAE = (torch.abs(ct_dose - pred) * head_mask).sum() / head_mask.sum()
                dist.all_reduce(MAE, op=torch.distributed.ReduceOp.SUM)
                MAE = MAE / dist.get_world_size()

                val_epoch_MAE.append(MAE.item())

                pred = pred.detach().cpu().numpy().squeeze()
                ct_dose = ct_dose.detach().cpu().numpy().squeeze()
                head_mask = head_mask.detach().cpu().numpy().squeeze()

            #if dist.get_rank() == 0:
            #    val_epoch_MAE = np.mean(val_epoch_MAE)
            #    print(f"epoch {epoch + 1} val MAE:", val_epoch_MAE)
            #
            #    if val_epoch_MAE < best_MAE:
            #        best_MAE = val_epoch_MAE
            #        torch.save(model.state_dict(),
            #                   os.path.join('trained_models', save_name, save_epoch, 'model_best_mae.pth'))
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_epoch_MAE,
            }
        
            save_epoch = f'{epoch + 1:04d}'
            torch.save(checkpoint,
                       os.path.join('trained_models', save_name, 'model_epoch' + str(save_epoch) + '.pth'))

                
                #print("pred:", np.min(pred), np.max(pred))
                #print("ct_dose:", np.min(ct_dose), np.max(ct_dose))
                
                #pred = pred.astype(np.uint8)
                #ct_dose = ct_dose.astype(np.uint8)
                
                #image = Image.fromarray(ct_dose)
                #image.save(f"{args.save_path}/gt/{case_num:03d}_Conv_dose_{slice_num:03d}.jpg")
                
                #image = Image.fromarray(pred)
                #image.save(f"{args.save_path}/pred/{case_num:03d}_sConv_dose_{slice_num:03d}.jpg")
                

'''
ct_dose_nifti = nib.Nifti1Image(ct_dose, affine=np.eye(4))  # Assuming identity affine for simplicity
nib.save(ct_dose_nifti, f"/panfs/jay/groups/1/bolanp1/lee03851/others/DoseDiff/trained_models/T1000_bs32_epoch1600_0413/{case_num:03d}_Conv_dose_{slice_num:03d}.nii.gz")

# Save pred as NIfTI
pred_nifti = nib.Nifti1Image(pred, affine=np.eye(4))  # Assuming identity affine for simplicity
nib.save(pred_nifti, f"/panfs/jay/groups/1/bolanp1/lee03851/others/DoseDiff/trained_models/T1000_bs32_epoch1600_0413/{case_num:03d}_sConv_dose_{slice_num:03d}.nii.gz")

head_nifti = nib.Nifti1Image(head_mask, affine=np.eye(4))  # Assuming identity affine for simplicity
nib.save(head_nifti, f"/panfs/jay/groups/1/bolanp1/lee03851/others/DoseDiff/trained_models/T1000_bs32_epoch1600_0413/{case_num:03d}_head_mask_{slice_num:03d}.nii.gz")
'''
