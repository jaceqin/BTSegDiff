import torchvision.transforms as transforms
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.utils import staple
import torchvision.utils as vutils
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.Thyroidloader import ThyroidDataset
from guided_diffusion.optloader import OPTDatset
from guided_diffusion import dist_util, logger
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
from PIL import Image
import torch as th
import time
import numpy as np
import random
import sys
import argparse
import os
from ssl import OP_NO_TLSv1
import nibabel as nib
import pandas as pd
from medpy.metric.binary import dc, jc, hd95
# import SimpleITK as sitk
from pandas import DataFrame
from visdom import Visdom
# viz = Visdom(port=8097)
sys.path.append(".")
np.set_printoptions(suppress=True)

seed = 10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def compute_uncer(pred_out):
    pred_out = th.sigmoid(pred_out)
    pred_out[pred_out < 0.001] = 0.001
    uncer_out = - pred_out * th.log(pred_out)
    return uncer_out


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),]
        transform_test = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_test, mode='Test')
        args.in_ch = 4

    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size, args.image_size)),]
        transform_test = transforms.Compose(tran_list)

        ds = BRATSDataset(args.data_dir, transform_test, test_flag=False)
        args.in_ch = 5

    elif args.data_name == 'OPT':
        tran_list = [transforms.Resize((args.image_size, args.image_size)),transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)

        ds = OPTDatset(args.data_dir, transform_train)
        args.in_ch = 4

    elif args.data_name=="Thyroid":
        tran_list = [transforms.Resize((args.image_size, args.image_size)),transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)

        ds = ThyroidDataset(args.image_path,args.mask_path,transform_train, test_flag=False)
        args.in_ch = 2


    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []

    state_dict = dist_util.load_state_dict(
        args.model_path, map_location="cuda")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
 # ***********************#
    dice = 0
    Jaccard = 0
    hd = 0
    dice_all = 0
    Jaccard_all = 0
    HD95_all = 0
    dice_allindices = 0

    dice_MCall = 0
    Jaccard_MCall = 0
    HD95_MCall = 0
    dice_MCallindices = 0

    dice_ensresall=0
    Jaccard_ensresall = 0
    HD95_ensresall = 0
    dice_ensresallindicesall = 0
    # ***********************#
    while len(all_images) * args.batch_size < args.num_samples:
        # should return an image from the dataloader "data"  b is no seg,m is seg ,path is seg path
        b, m, path = next(data)
        c = th.randn_like(b[:, :1, ...])
        gt_m = m
        img = th.cat((b, c), dim=1)  # add a noise channel$
        if args.data_name == 'ISIC':
            slice_ID = path[0].split("_")[-1].split('.')[0]
        elif args.data_name == 'BRATS':
            # slice_ID = path[0].split(
            #     "_")[3] + "_slice_" + path[0].split("_")[5].split('.nii')[0]  # BraTs2020
            slice_ID = path[0].split(
                "_")[2] + "_slice_" + path[0].split("_")[4].split('.nii')[0]  # BraTs2021
            # # slice_ID=path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]
            print("slice_ID is", slice_ID)
        elif args.data_name == 'Thyroid':
            slice_ID = path[0].split(".")[0].split("/")[2]
            print("slice_ID is", slice_ID)
        elif args.data_name == 'OPT':
            slice_ID = path[0].split(".")[0].split("/")[2]
            print("slice_ID is", slice_ID)
        logger.log("sampling...")
        

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []
        uncer_step = 2
        sample_outputs = []

        # this is for the generation of an ensemble of 5 masks.
################################################################################################################################
        m_array = gt_m.numpy()
        # m_array = np.where(m_array >= 1.0, 1, 0) #BraTs
        for i in range(args.num_ensemble):
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, cal_out, final = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                step=args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            sample_outputs.append(final)

            end.record()
            th.cuda.synchronize()
            # time measurement for the generation of 1 sample
            print('time for 1 sample', start.elapsed_time(end))

            co = th.as_tensor(cal_out)
            enslist.append(co)
            if m_array.sum() == 0:
                print("I jump")
                break
            if args.debug:
                if args.data_name == 'ISIC':
                    s = th.tensor(sample)[
                        :, -1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
                    o = th.tensor(org)[:, :-1, :, :]
                    c = th.tensor(cal).repeat(1, 3, 1, 1)
                    co = co.repeat(1, 3, 1, 1)
                elif args.data_name == 'BRATS':
                    s = th.tensor(sample)[:, -1, :, :].unsqueeze(1)
                    m = th.tensor(m.to(device='cuda:0'))[
                        :, 0, :, :].unsqueeze(1)
                    o1 = th.tensor(org)[:, 0, :, :].unsqueeze(1)
                    o2 = th.tensor(org)[:, 1, :, :].unsqueeze(1)
                    o3 = th.tensor(org)[:, 2, :, :].unsqueeze(1)
                    o4 = th.tensor(org)[:, 3, :, :].unsqueeze(1)
                    c = th.tensor(cal)
                elif args.data_name ==  'Thyroid':
                    s = th.tensor(sample)
                    # s=th.squeeze(s)
                    o = th.tensor(org)[:, :-1, :, :]
                    c = th.tensor(cal)
                    m = th.tensor(m.to(device='cuda:0'))[
                        :, 0, :, :].unsqueeze(1)
                elif args.data_name ==  'OPT':
                    s = th.tensor(sample)[
                        :, -1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
                    o = th.tensor(org)[:, :-1, :, :]
                    c = th.tensor(cal).repeat(1, 3, 1, 1)
                    co = co.repeat(1, 3, 1, 1)
                    m = th.tensor(m.to(device='cuda:0'))[
                        :, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
                    
                 
                # tup = (o1/o1.max(), o2/o2.max(), o3 /
                #        o3.max(), o4/o4.max(), m, s, c, co)   #BraTs
                tup=(o/o.max(),m,s,c,co) #Thyroid and OPT
                # tup=(o/o.max(),m,s,c,co) #OPT  

                # viz.image(visualize(o1), opts=dict(caption="input1"))
                # viz.image(visualize(o2), opts=dict(caption="input2"))
                # viz.image(visualize(o3), opts=dict(caption="input3"))
                # viz.image(visualize(o4), opts=dict(caption="input4"))
                # viz.image(visualize(m), opts=dict(caption="GT"))
                compose = th.cat(tup, 0)
            # save
            vutils.save_image(compose, fp=args.out_dir + "Validation_" +
                              str(slice_ID)+'_output'+str(i)+".jpg", nrow=1)
            # metrics
            # gt_img=nib.load(path[0])
            # gt_array=np.asarray(gt_img.get_fdata())
          
            co_array = co.cpu().detach().numpy()
            # co_array = np.where(co_array >= 1, 1, 0) #BraTS
            co_array = np.where(co_array[:,:1,::] >= 1, 1, 0) #OPT
            # co_array = np.where(co_array >=0.1, 1, 0) #Thy
            if co_array.sum() > 0:
                # dice
                dice = dc(co_array, m_array)
                dice_all = dice_all+dice
                # Jaccard
                Jaccard = jc(co_array, m_array)
                Jaccard_all = Jaccard_all+Jaccard
                # HD95
                hd = hd95(co_array, m_array)
                HD95_all = HD95_all+hd

                dice_allindices = dice_allindices+1
                # log
                logger.log("dice is", dice)
                logger.log("Jaccard is", Jaccard)
                logger.log("HD95 is", hd)
                logger.log("dice_all is", dice_all)
                logger.log("dice_allindices is", dice_allindices)
                logger.log('dice_mean is', dice_all / dice_allindices)
                logger.log('Jaccard_mean is',
                        Jaccard_all / dice_allindices)
                logger.log('HD95_mean is', HD95_all / dice_allindices)
                logger.log('\n')
            # vutils.save_image(m, fp = args.out_dir +"Validation_"+str(slice_ID)+'_seg'+str(i)+".jpg", nrow = 1)

#********************ensers**************
        if m_array.sum() != 0:
            ensres = staple(th.stack(enslist,dim=0)).squeeze(0) #merge five samples
            # ensres = th.sigmoid(ensres)
            # ensres = (ensres >= 1).float().cpu().numpy() #Brain
            ensres = (ensres >= 0.1).float().cpu().numpy() #Thy
            if ensres.sum() > 0 :
                # dice_fussion = dc(sample_return, m_array)
                # fusssion_all = fusssion_all+dice_fussion
                # fussion_indices = fussion_indices+1
                # dice
                dice_ensres = dc(ensres, m_array)
                dice_ensresall = dice_ensresall+dice_ensres
                # Jaccard
                Jaccard_ensres = jc(ensres, m_array)
                Jaccard_ensresall = Jaccard_ensresall+Jaccard_ensres
                # HD95
                hd_ensres = hd95(ensres, m_array)
                HD95_ensresall = HD95_ensresall+hd_ensres

                dice_ensresallindicesall = dice_ensresallindicesall+1
                # log
                logger.log("dice_ensres is", dice_ensres)
                logger.log("Jaccard_ensres is", Jaccard_ensres)
                # logger.log("HD95_ensres is", hd_ensres)
                logger.log("dice_ensresall is", dice_ensresall)
                logger.log("dice_ensresallindices is", dice_ensresallindicesall)
                logger.log('dice_ensresmean is', dice_ensresall / dice_ensresallindicesall)
                logger.log('Jaccard_ensresmean is',
                        Jaccard_ensresall / dice_ensresallindicesall)
                logger.log('HD95_ensresmean is', HD95_ensresall / dice_ensresallindicesall)
                logger.log('\n') 

                ensres = th.as_tensor(ensres)
                vutils.save_image(ensres, fp = args.out_dir +"Validation_"+str(slice_ID)+'_output_ens'+".jpg", nrow = 1)


################################################################################################################################
        # for i in range(uncer_step):
        #     sample, x_noisy, org, cal, cal_out,final = sample_fn(
        #         model,
        #         (args.batch_size, 3, args.image_size, args.image_size), img,
        #         step = args.diffusion_steps,
        #         clip_denoised=args.clip_denoised,
        #         model_kwargs=model_kwargs,
        #     )
        sample_return = th.zeros((1, 1, 256, 256))
        if m_array.sum() != 0:
            for index in range(995, 1000):
                uncer_out = 0
                for i in range(uncer_step):
                    # function p
                    uncer_out += sample_outputs[i]["all_model_outputs"][index]
                uncer_out = uncer_out / uncer_step
                uncer = compute_uncer(uncer_out).cpu()

                w = th.exp(th.sigmoid(
                    th.tensor((index + 1) / 5)) * (1 - uncer))
                                
                for i in range(uncer_step):
                    # function Y
                    sample_return += w * \
                        sample_outputs[i]["all_samples"][index].cpu(
                        )

            sample_return = th.sigmoid(sample_return)
            # sample_return = (sample_return >= 1).float().cpu().numpy() #Brain
            sample_return = (sample_return >= 1).float().cpu().numpy() #Thy
            ######################
            if sample_return.sum() > 0 and m_array.sum() > 0:
                # dice_fussion = dc(sample_return, m_array)
                # fusssion_all = fusssion_all+dice_fussion
                # fussion_indices = fussion_indices+1
                # dice
                dice_MC = dc(sample_return, m_array)
                dice_MCall = dice_MCall+dice_MC
                # Jaccard
                Jaccard_MC = jc(sample_return, m_array)
                Jaccard_MCall = Jaccard_MCall+Jaccard_MC
                # HD95
                hd_MC = hd95(sample_return, m_array)
                HD95_MCall = HD95_MCall+hd_MC

                dice_MCallindices = dice_MCallindices+1
                # log
                logger.log("dice_MC is", dice_MC)
                logger.log("Jaccard_MC is", Jaccard_MC)
                logger.log("HD95_MC is", hd_MC)
                logger.log("dice_MCall is", dice_MCall)
                logger.log("dice_MCallindices is", dice_MCallindices)
                logger.log('dice_MCmean is', dice_MCall / dice_MCallindices)
                logger.log('Jaccard_MCmean is',
                        Jaccard_MCall / dice_MCallindices)
                logger.log('HD95_MCmean is', HD95_MCall / dice_MCallindices)
                logger.log('\n')

                sample_return = th.as_tensor(sample_return)
                # viz.image(visualize(sample_return), opts=dict(
                #     caption="sampled output"))
                vutils.save_image(sample_return, fp=args.out_dir +
                                "Validation_"+str(slice_ID)+'_sample_return'+".jpg", nrow=1)
# *********************************************

# *****************************************************
def create_argparser():
    defaults = dict(
        data_name='BRATS',
        data_dir="",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5,  # number of samples in the ensemble
        gpu_dev="2",
        out_dir='./results/',
        multi_gpu=None,  # "0,1,2"
        debug=True,
        image_path="",
        mask_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
