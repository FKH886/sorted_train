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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def tokenize_dataset(dataset, tokenizer, context_length):
	train_dataset = load_dataset(path="wikitext", name="wikitext-103-raw-v1", split="train")
	dev_dataset = load_dataset(path="wikitext", name="wikitext-103-raw-v1", split="validation")
	tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
	tokenized_dataset = DatasetDict({
		'train': train_dataset,
		'valid': dev_dataset
	})
	tokenizer.pad_token = tokenizer.eos_token

	def tokenize(dataset):
		outputs = tokenizer(
			dataset["text"],
			truncation=True,
			max_length=context_length,
			return_overflowing_tokens=True,
			return_length=True,
		)
		input_batch = []
		for length, input_ids in zip(outputs['length'], outputs['input_ids']):
			if length == context_length:
				input_batch.append(input_ids)
		return {"input_ids": input_batch}
	print("Tokenizing dataset...")
	tokenized_datasets = tokenized_dataset.map(
		tokenize, batched=True, remove_columns=tokenized_dataset["train"].column_names
	)
	tokenized_datasets.save_to_disk("./tokenized_datasets")
def main(args):
	logging.basicConfig(filename=f'pretrainGPT2.log', filemode='w', level=logging.INFO,
	                    format='%(name)s - %(levelname)s - %(message)s')

	context_length = args.context_length
	logging.info("Loading dataset...")
	tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
	tokenizer.pad_token = tokenizer.eos_token
	tokenized_datasets = DatasetDict.load_from_disk("./tokenized_datasets_unsort_wiki103")
	logging.info("Loading model...")
	model_dir = args.model_dir
	model = GPT2LMHeadModel.from_pretrained("gpt2")
	# model_size = sum(t.numel() for t in model.parameters())
	# print(f"GPT-2 size: {model_size / 1024 ** 2:.1f}M parameters")
	# model.save_pretrained("./saved_model_large")
	data_collator = DataCollatorForSeq2Seq(tokenizer)
	batch_size = args.batch_size
	train_dataloader = DataLoader(
		tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
	)
	eval_dataloader = DataLoader(
		tokenized_datasets["valid"], shuffle=False, batch_size=batch_size, collate_fn=data_collator
	)

	optimizer = optim.AdamW(model.parameters(), lr=args.lr)
	num_epochs = args.num_epochs
	num_training_steps = num_epochs * len(train_dataloader)
	lr_scheduler = get_scheduler(
		"linear",
		optimizer=optimizer,
		num_warmup_steps=0,
		num_training_steps=num_training_steps,
	)
	model.to(device)
	progress_bar = tqdm(range(num_training_steps))
	logging.info("Training model...")
	model.train()
	dev_losses = []
	for epoch in range(num_epochs):
		print_gpu_memory()
		start_time = time.time()
		for batch in train_dataloader:
			batch = {k: v.to(device) for k, v in batch.items()}
			outputs = model(**batch, labels=batch['input_ids'])
			loss = outputs.loss
			loss.backward()
			optimizer.step()
			lr_scheduler.step()
			optimizer.zero_grad()
			progress_bar.update(1)
		end_time = time.time()
		logging.info(f"Epoch {epoch} finished in {(end_time - start_time)/60} minutes")
		logging.info(f'lr: {lr_scheduler.get_last_lr()[0]}')
		logging.info("Evaluating model...")
		losses = []
		for batch in eval_dataloader:
			batch = {k: v.to(device) for k, v in batch.items()}
			with torch.no_grad():
				outputs = model(**batch, labels=batch['input_ids'])
			losses.append(outputs.loss)
		# calculate perplexity
		perplexity = torch.exp(torch.stack(losses).mean())
		logging.info(f"Perplexity: {perplexity}")
		dev_losses.append(perplexity)
		if len(dev_losses) > 1 and dev_losses[-1] > dev_losses[-2]:
			logging.info("dev loss decreased once, skip saving model")
		elif len(dev_losses) > 2 and dev_losses[-1] > min(dev_losses) and dev_losses[-2] > min(dev_losses):
			logging.info("early stopping")
			break
		elif len(dev_losses) <=1 or dev_losses[-1] < min(dev_losses[:-1]):
			logging.info(f"Saving model to {model_dir}")
			model.save_pretrained(model_dir)
	print("Training finished!")
	print(dev_losses)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--num_epochs', type=int, default=3)
	parser.add_argument('--context_length', type=int, default=128)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--model_dir', type=str, default='./saved_model_finetunedGPT2_unsort_wiki103')
	args = parser.parse_args()
	main(args)
