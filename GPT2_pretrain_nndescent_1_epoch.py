from tqdm import tqdm  # progress bar
import numpy as np
import logging
import argparse
import torch
import time
from torch import nn
import torch.nn.functional as F
import evaluate
import subprocess
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, Dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, AutoConfig, DataCollatorForSeq2Seq, get_scheduler
import pynndescent
import pickle

logging.basicConfig(filename=f'GPT2_pretrain_nndescent_1_epoch.log', filemode='w', level=logging.INFO,
                    format='%(name)s - %(levelname)s - %(message)s')
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
    logging.info("No GPU available")

_start = time.time()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {device}")
context_length = 128
model_dir = '/scratch4/danielk/dwang119/saved_model_pretrainGPT2_nndescent_1_epoch'
batch_size = 128
data_dir = '/scratch4/danielk/dwang119/tokenized_embed_wiki103_'
logging.info(f"Loading model and tokenizer, time: {time.time() - _start} seconds")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
config = AutoConfig.from_pretrained(
                  "gpt2",
                  vocab_size=len(tokenizer),
                  n_ctx=context_length,
                  bos_token_id=tokenizer.bos_token_id,
                  eos_token_id=tokenizer.eos_token_id,
                  attention_mask=True,
                  pad_token_id=tokenizer.eos_token_id)
model = GPT2LMHeadModel(config)
model.to(device)
logging.info(f"Loading dataset from {data_dir}, time: {time.time() - _start}")
train_dataset = Dataset.load_from_disk(data_dir + 'train')
dev_dataset = Dataset.load_from_disk(data_dir + 'dev')
logging.info(f"Loading embedding_pca index, time: {time.time() - _start}")
start = time.time()
with open('/scratch4/danielk/dwang119/sorted_index/index_train_128.pkl', 'rb') as f:
    index_train = pickle.load(f)
index_train.prepare()

eval_dataloader = DataLoader(dev_dataset, batch_size=128)

def evaluation():
  losses = []
  model.eval()
  for batch in eval_dataloader:
    batch = torch.stack(batch['input_ids']).to(device)
    with torch.no_grad():
      outputs = model(batch, labels=batch)
    losses.append(outputs.loss)
  # calculate perplexity
  perplexity = torch.exp(torch.stack(losses).mean())
  logging.info(f"Perplexity: {perplexity}")
  return perplexity.item()

num_epochs = 1
cluster_size = 16
num_clusters = 16
batch_size = 128
check = True
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
num_training_steps = int(train_dataset.num_rows / num_clusters) * num_epochs
lr_scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_losses = []
dev_losses = []
pesudo_epoch = 0
print_gpu_memory()
logging.info(f"Start training, time: {time.time()}")
for epoch in range(num_epochs):
  start_time = time.time()
  start = time.time()
  progress_bar = tqdm(range(train_dataset.num_rows//num_clusters))
  train_loss = 0
  torch.cuda.empty_cache()
  print_gpu_memory()
  for i in range(len(train_dataset)//num_clusters):
    model.train()
    i = i * num_clusters
    data = train_dataset[i:i+num_clusters]['embedding_pca']
    neighbors = set(index_train.query(data, k=cluster_size)[0].flatten())
    batch = torch.tensor([train_dataset[int(i)]['input_ids'] for i in neighbors]).to(device)
    outputs = model(batch, labels=batch)
    if check:
      print_gpu_memory()
      check = False
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)
    torch.cuda.empty_cache()
    if (i + 1) % (train_dataset.num_rows // cluster_size // num_clusters) == 0:
      dev_losses.append(evaluation())
      pesudo_epoch += 1
      logging.info(f'Pesudo Epoch {pesudo_epoch} finished in {(start - time.time())/60} minutes')
      start = time.time()
      model.save_pretrained(model_dir)
      logging.info(f'lr: {lr_scheduler.get_last_lr()[0]}')
  model.save_pretrained(model_dir)
logging.info(f'Epoch {epoch} finished in {(start_time - time.time())/60/60} hours')
logging.info("Training finished!")