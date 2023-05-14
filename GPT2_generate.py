from transformers import GPT2TokenizerFast, GPT2LMHeadModel, AutoConfig, DataCollatorForSeq2Seq, get_scheduler
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("./saved_model_pretrainedGPT2_unsort_wiki103")
# model = GPT2LMHeadModel.from_pretrained("/Users/wangdixuan/Documents/school/SS "
#                                         "NLP/sorted_training/saved_model_pretrainGPT2_nndescent_1_epoch")
output = model.generate(tokenizer('I am a 15 year old boy ', return_tensors='pt')['input_ids'],
                        max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0]))
