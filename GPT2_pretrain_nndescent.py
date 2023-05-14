from tqdm import tqdm  # progress bar
import numpy as np
import logging
import argparse
import torch
import time
from torch import nn
import torch.nn.functional as F
import evaluate

import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, Dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, AutoConfig, DataCollatorForSeq2Seq, get_scheduler
import pynndescent
from my_utils import *



def main(args):
    logging.basicConfig(filename=f'pretrainGPT2_nndescent_regular.log', filemode='w', level=logging.INFO,
                        format='%(name)s - %(levelname)s - %(message)s')
    # get all the arguments

    model, tokenizer, device, train_dataset, dev_dataset, index_train, eval_dataloader = initialize(args)
    logging.info(f'Model loaded')
    print_gpu_memory()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * train_dataset.num_rows / args.batch_size

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps*2)

    logging.info("Training model...")
    train_losses = []
    dev_losses = []
    dev_perplexities = []
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = sorted_train(model, device, index_train, train_dataset, optimizer, lr_scheduler, args)
        train_losses.append(train_loss)
        if epoch == 0:
            print_gpu_memory()
        dev_loss, perplexity = my_evaluate(model, device, eval_dataloader)
        dev_losses.append(dev_loss)
        dev_perplexities.append(perplexity)
        if dev_losses[-1] == min(dev_losses):
            # torch.save(model.state_dict(), f'{args.model_dir}/best_model.pt')
            model.save_pretrained(f'{args.model_dir}/best_model/')
            logging.info(f"Best model saved, learning rate={lr_scheduler.get_last_lr()[0]}")
    logging.info(f"Training complete")
    logging.info(f"tain_losses={train_losses}")
    logging.info(f"dev_losses={dev_losses}")
    logging.info(f"dev_perplexities={dev_perplexities}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_dir', type=str, default='/scratch4/danielk/dwang119/saved_model_pretrainedGPT2_wiki103_nndescent_regular')
    parser.add_argument('--data_dir', type=str, default='/scratch4/danielk/dwang119/tokenized_embed_wiki103_')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--least_neighbors', type=int, default=16)
    args = parser.parse_args()
    main(args)
