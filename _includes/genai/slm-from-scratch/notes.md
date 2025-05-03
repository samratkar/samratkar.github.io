1. anything less than one billion. more performance with small size.
2. Steps
   1. our data set
   2. preprocessing
   3. Assembling the model architecture 
   4. training
   5. training with data augmentation
   6. 

## Step 1 : Our data set 
[Tiny Stories](https://arxiv.org/abs/2305.07759)
2 M : training rows
20,000 : test rows  
## Step 2 : Data preprocessing 
### Tokenization 
![](/assets/genai/slmfromscratch/transformer-block.png)
1. word based tokenization - word2vec, fasttext, glove, word2vec, BERT, GPT-2, T5, BART, RoBERTa, XLNet, ALBERT, DistilBERT, ELECTRA, Turing-NLG. 
   1. spelling mistakes will not be captured. 
   2. Vocabulary size is huge. 0.5 million words are there in the english language.
2. character based tokenization - #todo 
3. subword based tokenization - 
   1. Byte Pair Encoding (BPE) - BERT, GPT-2, T5, BART, RoBERTa, XLNet, ALBERT, DistilBERT, ELECTRA, Turing-NLG. 
   2. 
#### store all token ids in train.bin file 
This avoids RAM overload. 
![](/assets/genai/slmfromscratch/tokens.png)

![](/assets/genai/slmfromscratch/trainbin.png)

Train.bin would be created in batches

```python
!pip install tiktoken
import tiktoken
import os
import numpy as np
from tqdm.auto import tqdm

enc = tiktoken.get_encoding("gpt2")

# Some functions from https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py

## train.bin and test.bin created 
# subword token id and length 
def process(example):
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    out = {'ids': ids, 'len': len(ids)}
    return out

if not os.path.exists("train.bin"):
    tokenized = ds.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=8,
        )
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f'{split}.bin'
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        ## memory mapped array. 
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        ## batches 
        total_batches = 1024
        ## creating memory mapped array. 
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        ## save everything to disk
        arr.flush()
```

### Step 3 : Creating input output pairs. 
![](/assets/genai/slmfromscratch/input-output.png)
1. each time one token is generated.
2. **context size** - max number of tokens that LLM can look at before generating the next token.
3. **batch size** - backward propagation is done one batch at a time. having batches makes it convienent to train the model.
#### Input batch 
![](/assets/genai/slmfromscratch/batch.png)

#### Output batch 
![](/assets/genai/slmfromscratch/1inputout.png)

#### 4 tasks 
1. one --> day 
2. one day --> a 
3. one day a --> little 
4. one day a little --> girl 

For every row 4 input out prediction tasks are there. 

Why model understands that this cannot be predicted - **blue electrons eat fish**??
from random places the input and out batches are created. 
```python 
# Some functions from https://github.com/karpathy/nanoGPT/blob/master/train.py with slight modifications
# returns an input and output matrix. 
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('validation.bin', dtype=np.uint16, mode='r')
    # 4 random integers = positions that are random in the dataset. 
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # stacks x1, x2, x3, x4. block_size = 4 = context size 
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    # stacks y1, y2, y3, y4. block_size = 4 = context size
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        # cpu is not blocked and pinned to move to gpu easily. 
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

```

### Step 4 : Assembling the model architecture

***Embedding layer***
Every word has a meaning. It is the semantic notion of every word which is very important for the language model to capture. So, before passing to the transformer block, we need to convert the token ids to word embeddings to capture the meaning of the words. 

**Token embedding matrix** : Number of rows : vocabulary size (embedding dimension), column : context size 
The rows are indexed with the token positions. Token embedding matrix works as a lookup table.

- layer normalization becomes imporant - prevents the internal covariate shift. makes sure the distribution is similar. 



```python 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
import numpy as np
from tqdm.auto import tqdm
from contextlib import nullcontext
import os

## block1 - Layer normalization 
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
## block 2 - multi-head attention
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # y = context vector 
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection layer after dropout. 
        y = self.resid_dropout(self.c_proj(y))
        return y
## feedforward neural network. multi-layer perceptron.
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        ## expansion layer 
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        ## compression layer 
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
# Transformer block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # block 1 - layer normalization 
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        # block 2 - multi-head attention
        self.attn = CausalSelfAttention(config)
        # block 3 - layer normalization
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        # block 4 - feedforward neural network with dropout 
        self.mlp = MLP(config)
    def forward(self, x):
        # 1st shortcut 
        x = x + self.attn(self.ln1(x))
        # 2nd shortcut 
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        # multiple trasformer blocks
        for block in self.transformer.h:
            x = block(x)
        # layer normalization 
        x = self.transformer.ln_f(x)

        if targets is not None:
            ## neural network input = number of embeddings, and output as vocab size.
            logits = self.lm_head(x)
            ## computing the negative log likelihood loss.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens given a conditioning sequence.
        idx: Tensor of shape (B, T)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```

#### Parameter initialization 
![](/assets/genai/slmfromscratch/trainerparams.png)

#### Configuration 
```python
config = GPTConfig(
    vocab_size=50257, # use the tokenizer's vocab size
    block_size=128, # or whatever context size you're training with
    n_layer=6, # number of transformer blocks
    n_head=6, # number of multi-attention heads. each head compute different attention scores.
    n_embd=384, # embedding dimension. number of neurons in each layer.
    dropout=0.1, 
    bias=True
)

model = GPT(config)
```

### Step5 : Defining the loss function - Cross entropy loss

![](/assets/genai/slmfromscratch/loss.png)
![](/assets/genai/slmfromscratch/target.png)

```python 
## i have to run the model for a prescribed number of iterations. For 100 iterations, i will take 10 batches of data. 
def estimate_loss(model):
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out
```

### Step 6 : 
![](/assets/genai/slmfromscratch/step6.png)

```python 
# Training Config
import torch
from contextlib import nullcontext
## Hyper parameters
learning_rate = 1e-4 #more stable training, earlier 1e-4
max_iters = 20000 #increase from 25000
warmup_steps = 1000 #smoother initial train, earlier 100
min_lr = 5e-4 #lower rate, earlier 5e-4
eval_iters = 500 # increased from 100
batch_size = 32 # changed from 16, better gradient estimate
block_size = 128 #changed from 64, capture longer range dependencies

gradient_accumulation_steps = 32 # reduced from 50

device =  "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler

# How to use autocast https://wandb.ai/wandb_fc/tips/reports/How-To-Use-Autocast-in-PyTorch--VmlldzoyMTk4NTky
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
## note: float16 data type will automatically use a GradScaler
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.set_default_device(device)
torch.manual_seed(42)
```
### Step 7 : Training the model
### Gradient accumulation
![](/assets/genai/slmfromscratch/gradient%20accumulation.png)

Paremeter updates are done after every 32 batches. 

![](/assets/genai/slmfromscratch/learning%20rate.png)


```python
from torch.optim.lr_scheduler import LinearLR,SequentialLR, CosineAnnealingLR

##PUT IN WEIGHT DECAY, CHANGED BETA2 to 0.95
optimizer =  torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9) #weight decay for regularization

scheduler_warmup = LinearLR(optimizer, total_iters = warmup_steps) #Implement linear warmup
scheduler_decay = CosineAnnealingLR(optimizer,T_max = max_iters - warmup_steps, eta_min = min_lr) #Implement lr decay
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps]) #Switching from warmup to decay

# https://stackoverflow.com/questions/72534859/is-gradscaler-necessary-with-mixed-precision-training-with-pytorch
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
```

### Step 8 : Pre Training loop
1. for each interation, choose x and y. go through the data set choose random indices from data sets, and find input output pairs. Get the batch
2. Pass the entire batch to the model to get the logits.
3. calculate the loss using cross entropy loss, using logits and targets  y. 
4. back progpagation - compute the gradients.
5. Accumulate the gradients till we reach gradient_accumulation_steps. (32 in this case)
6. Update the parameters using the optimizer ADAMW.
7. Update learning rate using the scheduler. Every time parameters are updated.
8. Evaluate and save best model. Evaluation iterations used is 500. 

```python 
best_val_loss = float('inf')
best_model_params_path = "best_model_params.pt"
train_loss_list, validation_loss_list = [], []

# Ensure model is on the correct device
model = model.to(device)

# In your training loop
for epoch in tqdm(range(max_iters)):
    if epoch % eval_iters == 0 and epoch != 0:
        # Ensure estimate_loss uses the correct device
        losses = estimate_loss(model)
        print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        train_loss_list += [losses['train']]
        validation_loss_list += [losses['val']]

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), best_model_params_path)

    # Ensure X and y are on the correct device
    X, y = get_batch("train")
    X, y = X.to(device), y.to(device)
    # accumulate gradients
    with ctx:
        logits, loss = model(X, y)
        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
    # paraemeters are updatd only when gradient_accumulation_steps is reached.
    if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    scheduler.step()
```