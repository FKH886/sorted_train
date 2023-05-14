import logging
import subprocess
import torch
import time
from tqdm import tqdm  # progress bar
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, AutoConfig, DataCollatorForSeq2Seq, get_scheduler
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
import pickle
from torch.utils.data import DataLoader


def print_gpu_memory():
    """
    Print the amount of GPU memory used by the current process
    This is useful for debugging memory issues on the GPU
    """
    # check if gpu is available
    if torch.cuda.is_available():
        logging.info("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        logging.info("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        logging.info("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

        p = subprocess.check_output('nvidia-smi')
        logging.info(p.decode("utf-8"))
    else:
        logging.info("No GPU detected")


def initialize(args):
    logging.info("Loading model and tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=args.context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attention_mask=True,
        pad_token_id=tokenizer.eos_token_id)
    model = GPT2LMHeadModel(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    train_dataset = Dataset.load_from_disk(args.data_dir + 'train')
    dev_dataset = Dataset.load_from_disk(args.data_dir + 'dev')
    with open('/scratch4/danielk/dwang119/sorted_index/index_train_128.pkl', 'rb') as f:
        index_train = pickle.load(f)
    index_train.prepare()
    eval_dataloader = DataLoader(dev_dataset, batch_size=128, collate_fn=DataCollatorForSeq2Seq(tokenizer))
    return model, tokenizer, device, train_dataset, dev_dataset, index_train, eval_dataloader


def sorted_train(model,
                 device,
                 index,
                 train_dataset,
                 optimizer,
                 scheduler,
                 args):
    _start = time.time()
    model.train()
    seen = set()
    outliers = set()
    training_loss = 0
    count = 0
    for i, data in enumerate(train_dataset):
        if i in seen:
            continue
        data = data['embedding_pca']
        neighbors = index.query([data], k=args.batch_size)[0][0]
        neighbors = set(neighbors) - seen
        if len(neighbors) < args.least_neighbors:
            outliers.update(neighbors)
            continue
        batch = torch.tensor([train_dataset[int(j)]['input_ids'] for j in neighbors]).to(device)
        seen.update(neighbors)
        optimizer.zero_grad()
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        training_loss += loss.item()
        count += 1
    logging.info(f'sorted_train time: {time.time() - _start}')
    logging.info(f'now training on outliers: {len(outliers)}')
    for i in range(len(outliers)//args.batch_size):
        count += 1
        if (i+1)*args.batch_size > len(outliers):
            k = -1
        else:
            k = (i+1)*args.batch_size
        batch = torch.tensor([train_dataset[j]['input_ids'] for j in range(i*args.batch_size, k)]).to(device)
        optimizer.zero_grad()
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        training_loss += loss.item()
    logging.info(f'training_loss: {training_loss/count}, training_time: {time.time() - _start}')
    return training_loss/count


def my_evaluate(model, device, eval_dataloader):
    _start = time.time()
    model.eval()
    eval_loss = 0
    losses = []
    for batch in eval_dataloader:
        batch = {k: batch[k].to(device) for k in ['input_ids', 'attention_mask']}
        with torch.no_grad():
            outputs = model(**batch, labels=batch['input_ids'])
        eval_loss += outputs.loss.item()
        losses.append(outputs.loss)
        break
    logging.info(f'eval_loss: {eval_loss / len(eval_dataloader)}, eval_time: {time.time() - _start}')
    perplexity = torch.exp(torch.stack(losses).mean())
    logging.info(f"dev Perplexity: {perplexity}")
    return eval_loss / len(eval_dataloader), perplexity
