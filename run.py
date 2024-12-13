import os
import argparse
import logging
import sys

sys.path.append("..")

import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from models.bert_model import HMNeTNERModel
from processor.dataset import TwitterProcessor, WukongProcessor, MMPNERDataset
from modules.train import RETrainer, NERTrainer

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSOR = {
    'twitter15': TwitterProcessor,
    'twitter17': TwitterProcessor,
    "wukong": WukongProcessor
}
# select pretrained model path
PREtrainMODEL = {
    "bert-base-cased": "./pretrained_models/bert-base-cased",
    "bert-base-chinese": "./pretrained_models/bert-base-chinese",
    "bert-base-uncased": "./pretrained_models/bert-base-uncased"
}

DATA_PATH = {
    'twitter15': {
        # text data
        'train': './data/twitter15/train.txt',
        'dev': './data/twitter15/valid.txt',
        'test': './data/twitter15/test.txt',
        # {data_id : object_crop_img_path}
        'train_auximgs': './data/twitter15/twitter2015_train_dict.pth',
        'dev_auximgs': './data/twitter15/twitter2015_val_dict.pth',
        'test_auximgs': './data/twitter15/twitter2015_test_dict.pth'
    },

    'twitter17': {
        # text data
        'train': './data/twitter17/train.txt',
        'dev': './data/twitter17/valid.txt',
        'test': './data/twitter17/test.txt',
        # {data_id : object_crop_img_path}
        'train_auximgs': './data/twitter17/twitter2017_train_dict.pth',
        'dev_auximgs': './data/twitter17/twitter2017_val_dict.pth',
        'test_auximgs': './data/twitter17/twitter2017_test_dict.pth'
    },

    'wukong': {
        # text data
        'train': './data/wukong/train.txt',
        'dev': './data/wukong/valid.txt',
        'test': './data/wukong/test.txt',
        # {data_id : object_crop_img_path}
        'train_auximgs': "./data/wukong/train.dict",
        'dev_auximgs': "./data/wukong/dev.dict",
        'test_auximgs': "./data/wukong/test.dict"
    }
}

# image data
IMG_PATH = {
    'twitter15': './image_data/twitter15/twitter2015_images',
    'twitter17': './image_data/twitter17/twitter2017_images',
    "wukong": "./image_data/wukong/wukong-images"
}

# auxiliary images
AUX_PATH = {
    'twitter15': {
        'train': './image_data/twitter15/twitter2015_aux_images/train/crops',
        'dev': './image_data/twitter15/twitter2015_aux_images/val/crops',
        'test': './image_data/twitter15/twitter2015_aux_images/test/crops',
    },

    'twitter17': {
        'train': './image_data/twitter17/twitter2017_aux_images/train/crops',
        'dev': './image_data/twitter17/twitter2017_aux_images/val/crops',
        'test': './image_data/twitter17/twitter2017_aux_images/test/crops',
    },
    "wukong": {
        'train': './image_data/wukong/wukong_detect/train',
        'dev': './image_data/wukong/wukong_detect/dev',
        'test': './image_data/wukong/wukong_detect/test',
    }
}

Args = {
    "twitter15": {"bert_name": PREtrainMODEL["bert-base-uncased"],
                  "num_epochs": 50,
                  "batch_size": 8,
                  "lr": 3e-5,
                  "flood": 0.3},
    "twitter17": {"bert_name": PREtrainMODEL["bert-base-uncased"],
                  "num_epochs": 50,
                  "batch_size": 8,
                  "lr": 3e-5,
                  "flood": 0.5},
    "wukong": {"bert_name": PREtrainMODEL["bert-base-chinese"],
               "num_epochs": 10,
               "batch_size": 8,
               "lr": 1e-5,
               "flood": 0.6}
}


def set_seed(seed=1234):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='twitter15', type=str, help="The name of dataset.")
    parser.add_argument('--save_path', default="./ckpt_path/twitter15_ckpt_path", type=str,
                        help="save model at save_path")
    parser.add_argument('--flood', default=0.6, type=float, help="flood value")
    parser.add_argument('--num_epochs', default=10, type=int, help="num training epochs")  # 50  10
    parser.add_argument('--bert_name', default=PREtrainMODEL["bert-base-chinese"], type=str, help="Pretrained language "
                                                                                                  "model path")
    parser.add_argument('--save_result_path', default="./output/wukong", type=str,
                        help="")
    parser.add_argument('--save_pre_result_path', default="./case_study/wukong/pre.txt", type=str,
                        help="")
    parser.add_argument('--use_prompt', default=True, action='store_true')
    parser.add_argument('--use_similar', default=True, action='store_true')
    parser.add_argument('--cross_fusion', default=True, action='store_true')
    parser.add_argument('--do_train', default=True, action='store_true')
    parser.add_argument('--only_test', default=False, action='store_true')
    parser.add_argument('--batch_size', default=8, type=int, help="batch size")  # 8  16
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")  # 3e-5  1e-5

    parser.add_argument('--CUDA_VISIBLE_DEVICES', default=0, type=int, help="")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=3, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=1234, type=int, help="random seed, default is 1")  # 1234
    parser.add_argument('--prompt_len', default=4, type=int, help="prompt length")
    parser.add_argument('--prompt_dim', default=800, type=int, help="mid dimension of prompt project layer")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--write_path', default=None, type=str,
                        help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--max_seq', default=147, type=int)
    parser.add_argument('--ignore_idx', default=0, type=int)

    parser.add_argument('--encoder_v', default='resnet101', help="")
    parser.add_argument('--v_models_path', default='D:/zyk/pretrainModel/resnet/resnet101-63fe2227.pth', help="")
    # parser.add_argument('--t_models_path', default='D:/zyk/pretrainModel/ar-crawl-fasttext-300d-1M', help="")

    # auto args
    args = parser.parse_args()
    dataset_name = args.dataset_name
    args.save_path = "./ckpt_path/" + dataset_name + "_ckpt_path"
    args.bert_name = Args[dataset_name]["bert_name"]
    args.save_result_path = "./output/" + dataset_name
    args.save_pre_result_path = "./case_study/" + dataset_name + "/pre.txt"
    args.batch_size = Args[dataset_name]["batch_size"]
    args.lr = Args[dataset_name]["lr"]
    args.flood = Args[dataset_name]["flood"]

    with open(args.save_result_path, mode="a", encoding="UTF-8") as f:
        f.write(f"{args.num_epochs} epoch, {args.lr}, {args.batch_size}, {args.flood}, similar={args.use_similar}, "
                f"fusion={args.cross_fusion}\n")

    data_path, img_path, aux_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name], AUX_PATH[
        args.dataset_name]

    model_class, Trainer = HMNeTNERModel, NERTrainer
    data_process, dataset_class = (PROCESSOR[args.dataset_name], MMPNERDataset)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    set_seed(args.seed)  # set seed, default is 1
    if args.save_path is not None:  # make save_path dir
        # args.save_path = os.path.join(args.save_path, args.dataset_name+"_"+str(args.batch_size)+"_"+str(
        # args.lr)+"_"+args.notes)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    print(args)
    logdir = "new_logs/" + args.dataset_name + "_" + str(args.flood) + "_" + str(args.use_similar) + "_" + str(
        args.cross_fusion) + args.notes
    writer = SummaryWriter(logdir=logdir)
    # writer = None

    if not args.use_prompt:
        img_path, aux_path = None, None

    processor = data_process(data_path, args.bert_name, args)

    train_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq,
                                  sample_ratio=args.sample_ratio, mode='train', args=args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)

    dev_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, mode='dev', args=args)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, mode='test', args=args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 pin_memory=True)

    label_mapping = processor.get_label_mapping()

    label_list = list(label_mapping.keys())
    model = HMNeTNERModel(label_list, args).to(args.device)

    trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model,
                      label_map=label_mapping, args=args, logger=logger, writer=writer)

    if args.do_train:
        # train
        trainer.train()
        # test best model
        args.load_path = os.path.join(args.save_path, 'best_model.pth')
        trainer.test()

    if args.only_test:
        # only do test
        args.load_path = os.path.join(args.save_path, 'best_model.pth')
        trainer.test()

    torch.cuda.empty_cache()
    # writer.close()


if __name__ == "__main__":
    main()
