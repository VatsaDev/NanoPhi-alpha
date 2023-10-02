import time

out_dir = '/content/drive/MyDrive/Model' # for google colab, for regular use, set value to 'out-chat'
eval_interval = 5
eval_iters = 200
wandb_log = False # feel free to turn on
wandb_project = 'gpt2finetune'
wandb_run_name = 'ft-' + str(time.time())

dataset = ''
init_from = 'gpt2' # good for google colab

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

batch_size = 1
gradient_accumulation_steps = 32
max_iters = 200

# finetune at constant LR
learning_rate = 3e-4
decay_lr = False
