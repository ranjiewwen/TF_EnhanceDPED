#!/usr/bin/env python
# coding: utf-8

import sys
import argparse
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

## windows platform
experiment_name = os.path.splitext(__file__.split('\\')[-1])[0]
PROJECT_PATH="F:\\ranjiewen\\TF_EnhanceDPED"

## ubuntu platform
# experiment_name = os.path.splitext(__file__.split('/')[-1])[0]
# PROJECT_PATH="/home/rjw/desktop/graduation_project/TF_EnhanceDPED"

# specifying default parameters
def process_command_args(arguments):

    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Tensorflow Image Enhancement DPED Dataset Training")

    ## Path related arguments
    parser.add_argument('--exp_name', type=str, default=experiment_name, help='experiment name')
    parser.add_argument('--dataset_dir',type=str,default=os.path.join(PROJECT_PATH,'data','dped'),help='the root path of dataset')
    parser.add_argument('--checkpoint_dir',type=str,default=os.path.join(PROJECT_PATH,"experiments"),help='the path of ckpt file')
    parser.add_argument('--tesorboard_logs_dir',type=str,default = os.path.join(PROJECT_PATH,"experiments","logs"),help='the path of tensorboard logs')
    parser.add_argument('--pretrain_weights',type=str,default = os.path.join(PROJECT_PATH,"data","vgg_models","imagenet-vgg-verydeep-19.mat"))

    ## models retated argumentss
    parser.add_argument('--save_visual_result',type=str2bool,default=True, help="whether to save visual result file ")
    parser.add_argument('--save_ckpt_file', type=str2bool, default=True,
                        help="whether to save trained checkpoint file ")
    parser.add_argument('--split', type=str, default='train',help="choose from train/val/test/trainval")

    ## dataset related arguments
    parser.add_argument('--dataset',default='iphone',type=str,choices=["iphone", "sony", "blackberry"],help='datset choice')
    parser.add_argument('--patch_width',type=int,default=100,help='train patch width')
    parser.add_argument('--patch_height',type=int,default=100,help='train patch height')
    parser.add_argument('--patch_size',type=int,default=100*100*3,help='train patch size')

    ## train related arguments
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--train_size',type=int,default=3000) # 3000
    parser.add_argument('--test_size',type=int,default=300)  # 300
    parser.add_argument('--w_content',type=float,default=2)
    parser.add_argument('--w_color',type=float,default=0.5)
    parser.add_argument('--w_texture',type=float,default=5)
    parser.add_argument('--w_tv',type=float,default=2000)
    parser.add_argument('--eval_step',type=int,default=100) #100
    parser.add_argument('--summary_step',type=int,default=2)

    ## optimization related arguments
    parser.add_argument('--learning_rate',type=float,default=5e-4,help='init learning rate')
    parser.add_argument('--iter_max',type=int,default=96000,help='the maxinum of iteration')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=4e-5)

    args = parser.parse_args(arguments)
    return args


def process_test_model_args(arguments):

    phone = "iphone"
    dped_dir = os.path.join(PROJECT_PATH,'data','dped')
    test_subset = "small"
    iteration = "all"
    resolution = "orig"
    use_gpu = "true"

    for args in arguments:

        if args.startswith("model"):
            phone = args.split("=")[1]

        if args.startswith("dped_dir"):
            dped_dir = args.split("=")[1]

        if args.startswith("test_subset"):
            test_subset = args.split("=")[1]

        if args.startswith("iteration"):
            iteration = args.split("=")[1]

        if args.startswith("resolution"):
            resolution = args.split("=")[1]

        if args.startswith("use_gpu"):
            use_gpu = args.split("=")[1]

    if phone == "":
        print("\nPlease specify the model by running the script with the following parameter:\n")
        print("python test_model.py model={iphone,blackberry,sony,iphone_orig,blackberry_orig,sony_orig}\n")
        sys.exit()

    return phone, dped_dir, test_subset, iteration, resolution, use_gpu
