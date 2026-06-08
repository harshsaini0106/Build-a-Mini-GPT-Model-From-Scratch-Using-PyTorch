import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from transformer_block import Block

corpus=[
    "hello friends how are you",
"my name is harsh",
"i studied in delhi college",
"that was in the dwarka" ,
"i will play cricket",
"my favourite game is wrestling" ,
"i love to running" ,
"i started my gen ai course", 
"today i learn llm models" ,
"in which i learn tokenizer"
]

corpus= [s+" <END>" for s in corpus]
text=" ".join(corpus)
# print(text)
 
#convert to words
words=list(set(text.split()))
# print(words)
vocab_size=len(words)
# print(vocab_size) #38

#give unique id to every words
word2inx={w:i for i, w in enumerate(words)}
# print(word2inx)

idx2word={i: w for w, i in word2inx.items()}

#byte pair encoding(BPE) use in modern LLMs that do these 2 steps

#convert tokens to tensors
data=torch.tensor([word2inx[w] for w in text.split()],dtype=torch.long)
# print(data)
# print(len(data)) #58

block_size=6 #contex length
embedding_dim=32
n_heads=2
n_layers=2
lr=1e-3
epochs=1500

def get_batch(batch_size=16):
    ix=torch.randint(len(data)-block_size,(batch_size,)) #58-6 (0-51),16
    # 16 sequences = [4,2,31,...] #total 16
    # 4 (4,5,6,7,8,9) total 6 becoz block size =6
    x=torch.stack([data[i:i+block_size]for i in ix]) 
    y=torch.stack([data[i+1:i+block_size+1]for i in ix])
    #x= [token12,23,14,15,16,17]
    #y= [token13,14,15,16,17,18]
    return x,y

class Tinygpt(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding=nn.Embedding(vocab_size,embedding_dim) #38,32
        self.position_embedding=nn.Embedding(block_size,embedding_dim) #6,32
        self.blocks=nn.Sequential(*[Block(embedding_dim,block_size,n_heads)for _ in range(n_layers)])

        self.ln_f=nn.LayerNorm(embedding_dim)
        self.head=nn.Linear(embedding_dim,vocab_size)

    def forward(self,idx,targets=None):
        B,T=idx.shape #B= batch_size t=  block_size (16,6)
        tok_emb=self.token_embedding(idx) #16,6,32
        pos_emb=self.position_embedding(torch.arange(T,device=idx.device)) #16,6,38
        x=tok_emb+pos_emb
        x=self.blocks(x)
        x=self.ln_f(x)
        logits=self.head(x) # raw prediction
        loss=None
        if targets is not None:
            B,T,C=logits.shape #16,6,38
            loss=F.cross_entropy(logits.view(B*T,C),targets.view(B*T))
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-block_size:]
            logits, _ =self(idx_cond)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            next_idx=torch.multinomial(probs,1)
            idx=torch.cat((idx,next_idx),dim=1)
        return idx
    
model=Tinygpt()
optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
for step in range(epochs):
    xb,yb=get_batch()
    logits,loss=model(xb,yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 300==0:
        print(f"step {step},loss={loss.item():4f}")


context=torch.tensor([[word2inx["harsh"]]],dtype=torch.long)
out=model.generate(context,max_new_tokens=10)

print(" ".join(idx2word[int(i)]for i in out[0]))