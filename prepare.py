import os
import random
import requests
import tiktoken
import numpy as np

train_ids=[]
val_ids=[]
enc = tiktoken.get_encoding("gpt2")

chunk_no=0
def download_file(url, output_dir):
  global chunk_no
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  
  response = requests.get(url, stream=True)
  if response.status_code == 200:
    for chunk in response.iter_content(chunk_size=104857600):
      chunk_no=chunk_no+1
      output_filename = os.path.join(output_dir, f'{chunk_no}-dataset.txt')
      with open(output_filename, 'wb') as f:
        f.write(chunk)
        print(f"made chunk {chunk_no}")
    print("downloaded and chunked dataset, proceeding to tokenizing...")
  else:
    print('Error downloading file:', response.status_code)

download_file('https://huggingface.co/VatsaDev/unagami/resolve/main/data.txt', 'output')

train_len = 0
val_len = 0
train_no = 0
val_no = 0
for filename in os.listdir('output'): #blocks are chosen randomly from the text, more of a seamless train val split
  if filename.endswith('.txt'):
    train_or_val = random.randint(0, 9)
    if train_or_val <= 8:
      with open(f'output/{filename}', 'r') as f:
        data = f.read()
      train_ids = enc.encode_ordinary(data)
      train_len = train_len+len(train_ids)
      train_ids = np.array(train_ids, dtype=np.uint16)
      train_no = train_no+1
      train_ids.tofile(os.path.join(os.path.dirname(__file__), f'train{train_no}.bin'))
      print(f"train has {train_len} tokens")
      train_ids = []
    if train_or_val > 8:
      with open(f'output/{filename}', 'r') as f:
        data = f.read()
      val_ids = enc.encode_ordinary(data)
      val_len = val_len+len(val_ids)
      val_ids = np.array(val_ids, dtype=np.uint16)
      val_no = val_no+1
      val_ids.tofile(os.path.join(os.path.dirname(__file__), f'val{val_no}.bin'))
      print(f"val has {val_len} tokens")
      val_ids = []

# data loader
dataset = ''
data_dir = os.path.join('data', dataset)
total_train_data=[10] # just keeping arrays not empty
total_val_data=[10]
total_train_data=np.array(total_train_data, dtype=np.uint16)
total_val_data=np.array(total_val_data, dtype=np.uint16)
total_train_data.tofile('/content/unagami/data/traintotal.bin')
total_val_data.tofile('/content/unagami/data/valtotal.bin')
total_train_data=np.memmap(os.path.join(data_dir, 'traintotal.bin'), dtype=np.uint16, mode='r')
total_val_data=np.memmap(os.path.join(data_dir, 'valtotal.bin'), dtype=np.uint16, mode='r')

def concat_bins():
    global total_val_data
    global total_train_data
    for filename in os.listdir('data'):
      if filename.endswith('.bin'):
        if filename[:3] == 'val':
            # Val files
            print(f"concat {filename}")
            val_data = np.memmap(os.path.join(data_dir, filename), dtype=np.uint16, mode='r')
            total_val_data = np.concatenate([total_val_data, val_data])
            del val_data
            total_val_data.tofile('/content/unagami/data/valtotal.bin')
        else:
            # Train files
            print(f"concat {filename}")
            train_data = np.memmap(os.path.join(data_dir, filename), dtype=np.uint16, mode='r')
            total_train_data = np.concatenate([total_train_data, train_data])
            del train_data
            total_train_data.tofile('/content/unagami/data/traintotal.bin')
    print("concat over")

concat_bins()
