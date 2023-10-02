import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import requests

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = '/content/drive/MyDrive/Model' # finetuned model directory
num_samples = 1 # no samples. 1 for 1 chat at a time
max_new_tokens = 150
ans_long=True
temperature = 0.85 
top_k = 10 # retain only the top_k most likely tokens, clamp others to have 0 probability
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
context="<system>You are an AI assistant named UNAGAMI, designed to help users<endOfText>"
exec(open('configurator.py').read()) # overrides from command line
# -----------------------------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def download_ckpt(url):
  response = requests.get(url)
  if response.status_code == 200:
    with open('ckpt.pt', 'wb') as f:
      f.write(response.content)
  else:
    print('Error downloading file:', response.status_code)

# gets model
# init from a model saved in a specific directory
if init_from == 'huggingface':
  if os.path.isfile('ckpt.pt'):
    # init from huggingface model
    ckpt_path = 'ckpt.pt'
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict) 
  else:
    # init from huggingface model
    download_ckpt('https://huggingface.co/VatsaDev/unagami/resolve/main/ckpt.pt')
    ckpt_path = 'ckpt.pt'
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict) 
elif init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# gpt-2 encodings
print("loading GPT-4 encodings...")
enc = tiktoken.encoding_for_model("gpt-4")
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)


def remov_sys(text):
  tok_index = text.find("<system>", text.find("<system>") + 1)
  if tok_index == -1:
    return text
  else:
    return text[:tok_index]

def respond(input, samples): # generation function
    if ans_long == True:
      max_new_tokens=150
    else:
      max_new_tokens=50
    x = (torch.tensor(encode(input), dtype=torch.long, device=device)[None, ...]) 
    with torch.no_grad():
        with ctx:
            for k in range(samples):
                generated = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                output = decode(generated[0].tolist())

                # gen sample
                #print("--------------------")
                #print(output)
                #print("--------------------")
              
                # sanitation
                # replace context
                out = output.replace(input,'')
                # remove any extra system response
                out=out.partition('<system>')
                # remove any human response
                out =  out[0].partition('<human>')
                # if the bot has anything left afterwards, the endOfText token is put to use
                output_text =  out[0].rpartition('<endOftext>')
                output_text = out[0] + out[1]
                # label removing
                output_text = output_text.replace('<human>',' ')
                output_text = output_text.replace('<bot>',' ')
                output_text = output_text.replace('<endOfText>',' ')
                return output_text

# chat loop
while True:
    # get input from user
    start_input = input('User: ')
    start = '<human>'+start_input+'<endOfText><bot>'
    context=context+start
    
    out = respond(context, num_samples)
  
    context=context+out+'<endOfText><system>You are an AI assistant named UNAGAMI, designed to help users<endOfText>'
  
    print('Bot: '+ out)
