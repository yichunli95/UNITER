from collections import defaultdict
import argparse
import math
import os
from time import time
from tqdm import tqdm
import torch
from os.path import exists, join
import json
import csv
from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.save import ModelSaver, save_training_meta
from utils.const import IMG_DIM, IMG_LABEL_DIM, BUCKET_SIZE
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from optim.misc import build_optimizer
from optim import get_lr_sched
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F

from data_mlm import create_dataloaders
from data import (PrefetchLoader, MetaLoader)
from model.pretrain import UniterForPretraining
from pytorch_pretrained_bert import BertTokenizer

def main(opts):
    device = torch.device("cuda")
    set_random_seed(opts.seed)

    val_dataloaders, _ = create_dataloaders('nlvr2/img_db/nlvr2_dev',
        opts.val_datasets, False, opts)
    test_dataloaders, _ = create_dataloaders('nlvr2/img_db/nlvr2_test',
        opts.test_datasets, False, opts)
    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}
    model = UniterForPretraining.from_pretrained(
        opts.model_config, checkpoint,
        img_dim=IMG_DIM, img_label_dim=IMG_LABEL_DIM)
    model.to(device)
    model.train()
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    optimizer.zero_grad()
    optimizer.step()
    validate(model, val_dataloaders, 'val')
    validate(model, test_dataloaders, 'test')

def validate(model, dataloaders, setname):
    model.eval()
    for task, loader in dataloaders.items():
        LOGGER.info(f"validate on {task} task")
        if task.startswith('mlm'):
            val_log = validate_mlm(model, loader, setname)
        else:
            raise ValueError(f'Undefined task {task}')
        val_log = {f'{task}_{k}': v for k, v in val_log.items()}
        TB_LOGGER.log_scaler_dict(
            {f'valid_{task}/{k}': v for k, v in val_log.items()})
    model.train()

@torch.no_grad()
def validate_mlm(model, val_loader, setname):
    #cpu_device = torch.device("cpu")
    LOGGER.info("start running MLM validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    st = time()
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
    # fn = f'/content/mlm_{setname}_predictions.csv'
    fn = f'/content/mlm_{setname}_predictions_text_only.csv'
    with open(fn, 'w') as f:
        cw = csv.writer(f)
        cw.writerow(['actual', 'predict'])
        #print(len(val_loader))
        for i, batch in tqdm(enumerate(val_loader)):

            scores = model(batch, task='mlm', compute_loss=False)
            # What the masked words originally were
            labels = batch['txt_labels']
            labels = labels[labels != -1]
            loss = F.cross_entropy(scores, labels, reduction='sum')
            val_loss += loss.item()
            # What the masked words predictions were
            pred = scores.max(dim=-1)[1]
            n_correct += (pred == labels).sum().item()
            n_word += labels.numel()
            actual_words = tokenizer.convert_ids_to_tokens(labels.tolist())
            pred_words = tokenizer.convert_ids_to_tokens(pred.tolist())
            assert len(actual_words) == len(pred_words)
            for actual, pred in zip(actual_words, pred_words):
                cw.writerow([actual, pred])
    tot_time = time()-st
    val_loss /= n_word
    acc = n_correct / n_word
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_word/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}")
    return val_log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    # NOTE: train tasks and val tasks cannot take command line arguments
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')

    parser.add_argument("--model_config", type=str,
                        help="path to model structure config json")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="path to model checkpoint (*.pt)")

    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size", default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size", default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adamw',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=2.0, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=10000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for.")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=0,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_false', help="pin memory")

    # can use config files
    parser.add_argument('--config', required=True, help='JSON config files')

    args = parse_with_config(parser)

    if exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
