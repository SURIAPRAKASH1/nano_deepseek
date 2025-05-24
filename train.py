import importlib.util
import torch
import torch.optim as optim

from models.config.default import DeepSeekConfig
from models.deepseek_v3.transformer import DeepSeekTransformer

from collections import defaultdict
import sys
import importlib
import time 
from dataclasses import dataclass, field


# what's the current device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('current device -->', device)
print(" ")

# tokenizer
print("importing tokenizer from hugging_face...")
print(" ")


if importlib.util.find_spec('transformers'):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
    vocab_size = tokenizer.vocab_size
    print('vocab_size', vocab_size)
    print(" ")

else:
    print('We need DeepSeek-v3 tokenizer from transformers lib so install') 
    sys.exit(1)
    

print("Tokenizing the Dataset")
print(" ")
# lyrics file 
text = open("All_eminem_songs.txt", 'r').read()
tokens = tokenizer.encode(text)
print(f'{len(text)} words get tokenized to {len(tokens)} tokens')
print(" ")

print("Splitting train and dev dataset")
print(" ")
# splitting data into train and dev set's
n = int(len(tokens) * 0.9)
train_data = tokens[:n]      # 90%
dev_data = tokens[n:]        # 10%
print(f"train data {len(train_data)} tokens\ndev data {len(dev_data)} tokens")
print(" ")

# this what i can came up with to update model configuration from cml args
exec(open("configurator.py", 'r').read())

model_args = DeepSeekConfig()

cmla_configs = globals()["cml_updated_args"]
if len(cmla_configs) > 0:
    verbose_re_configs = {k: v for k, v in cmla_configs.items() if k != 'verbose'}
    if len(verbose_re_configs) > 0:
        model_args = DeepSeekConfig(**verbose_re_configs)
        if cmla_configs['verbose']:
          print("Updated model Arguments")
          print(model_args)
          print(" ")


# randomly get sample of batch of tokens
def get_batch(split, device):
  
  data = train_data if split == 'train' else dev_data
  xi = torch.randint(len(data) - model_args.block_size, (model_args.batch_size,))
  x = torch.tensor([data[i: i + model_args.block_size] for i in xi])
  y = torch.tensor([data[i + 1: i + model_args.block_size + 1 ] for i in xi])

  # for efficient gpu performence
  if device != 'cpu':
    x = x.pin_memory()              # by pinning make sure tensor ain't non pageble (only live in ram)
    y = y.pin_memory()
    x = x.to(device, non_blocking = True)
    y = y.to(device, non_blocking = True)
  else:
    x = x.to(device)
    y = y.to(device)

  return x, y

X, Y = get_batch('train', device)
# How transfomer see tokens and learn from it
# for single sequence . here i cut the seq for visualization
t = model_args.block_size // 5  if model_args.block_size // 10  <= 5 else 5
print("-----------------------------------HOW TRANSFORMER SEE TOKENS AND LEARN FROM IT---------------------------------")
for i in range(t):
  t_input = X[0, : i+1].tolist()
  t_pred = Y[0, i].tolist()
  print(f"Input: {t_input}, Have to predict: {t_pred}")
  print(f"Input: {tokenizer.decode(t_input)}, Have to predict: {tokenizer.decode(t_pred)}")
  print(' ') 


# Hyper parameters for training
steps: int = 5000           # How many steps we want to trian our model
eval_iters: int = 200       # When estimating a loss How many batches we should be consider
eval_step: int = 500        # evaluate loss once in a while
lr: float = 1e-4             # learning rate
min_lr: float = 1e-5          
beta1: float = 0.9
beta2: float = 0.95
weight_decay: float = 1e-1   
warmup_iters: int = 200    # will increase lr then start to decay from here 

@torch.no_grad()
def estimate_loss(model):
  model.eval()              # model in eval mode bro .....

  out = {}
  for split in ['train', 'dev']:
    losses = torch.zeros(eval_iters)

    for i in range(eval_iters):
      X, Y = get_batch(split, device = device)
      _, loss = model(X, Y)
      losses[i] = loss

    # take average over batches
    out[split] = losses.mean()

  model.train()
  return out

import math
def get_lr(it):
  # so we gradually increasing learning rate
  if it < warmup_iters:
    return  lr * (it + 1) / (warmup_iters + 1)
    
  # starting to decaying the learning rate using cosine
  else:
    decay_ratio = (it - warmup_iters)/ (steps - warmup_iters)
    assert 0 <= decay_ratio <=1
    coeff = 0.5 * ( 1.0 + math.cos( math.pi * decay_ratio))
    return  min_lr + coeff * (lr - min_lr)    # we make sure learning rate shouldn't 0 (but we wanna decrease)

print("initiating a model ...")
print(" ")
model = DeepSeekTransformer(model_args).to(device)

# AdamW (decoubled weight decay)
optimizer = optim.AdamW(model.parameters(), lr = lr, betas= (beta1, beta2), weight_decay= weight_decay)
scaler = torch.amp.GradScaler(device = device)

# loss stacks
@dataclass
class Loss:
  main_loss: list = field(default_factory= list)
  mtp_loss: list = field(default_factory= list)

gb_lossi = defaultdict(Loss)  

print("start training a model ...")
print(" ")
# Optimization loop

start = time.time()
for step in range(steps):
  # get batch of sample data from training dataset
  X, Y = get_batch('train', device)
  optimizer.zero_grad()

  # 1. FORWARD PASS AND COMPUTE LOSS

  # enable auto mixed percision. it's converts dtype to F16/BF16 whenever possible.
  with torch.amp.autocast(device_type= device, dtype= torch.bfloat16):
    _, loss = model(X, Y)

  # 2. BAKWARD PASS
  # scale the loss then do back-ward pass
  # cause computing loss in F16/BF16 dtype (if we) we get very small loss. if compute grad for that we will get vanishing gradients
  # so what's the solution scale the loss then compute gradients, when updating params scale down else explode
  scaler.scale(loss).backward()

  # grad clip
  scaler.unscale_(optimizer) 
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

  # 3.UPDATE PARAMETERS 
  scaler.step(optimizer)
  scaler.update()
  optimizer.defaults['lr'] = get_lr(step) 

  # estimate loss once in a while
  if step % eval_step == 0 or step  == steps - 1:
    out = estimate_loss(model)

    # for plotting loss curve
    gb_lossi['train'].main_loss.append(out['train'][0].item())
    gb_lossi['train'].mtp_loss.append(out['train'][1].item())

    gb_lossi['dev'].main_loss.append(out['dev'][0].item())
    gb_lossi['dev'].mtp_loss.append(out['dev'][1].item())


    print(f"step {step}/{steps}: train: main_loss {out['train'][0].item()} mtp_loss {out['train'][1].item()}, dev: main_loss {out['dev'][0].item()} mtp_loss {out['dev'][1].item()}")

print(" ")
print("training is complete ....")
print(" ")
end = time.time()
print("Training time %.2f " % ((end - start)/60), "Minutes")
print(" ")

# SAMPLING 
# encode string to get tokens
print("sampling from model ...")
print(" ")

prompt = """no more games, i'am change what you call rage"""
encoded_tokens =  torch.tensor([tokenizer.encode(prompt)], device= device) # (B, T) 

# sampling from model
model.eval()
generated_tokens =  model.generate(encoded_tokens, max_tokens= 100, temperature= 0.8, top_k= 10000)

# decode tokens to get string format 
result = tokenizer.decode(generated_tokens[0].tolist(), skip_special_tokens= True)
print(result)
