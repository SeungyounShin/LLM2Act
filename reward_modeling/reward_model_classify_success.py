import torch
import json 
from reward_model import RewardModel
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
from utils import *
from tqdm import tqdm

# for reward creation 
tokenizer = LlamaTokenizer.from_pretrained("/home/seungyoun/stanford_alpaca/ckpt/7B/tokenizer")
model = LlamaForCausalLM.from_pretrained("/home/seungyoun/stanford_alpaca/ckpt/7B/llama-7b")

# create an instance of the model

# load the trained weights
if tokenizer.pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
    )
    
tokenizer.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    }
)

model = RewardModel(model)
model.load_state_dict(torch.load('ckpt/reward_model.pt'))


# set the model to evaluation mode
model.eval()
print(f'model successfully loaded!')

# load demos
results = list()
demo_path = '/home/seungyoun/LLM2Act/demo_files/history_selfplay.json'
with open(demo_path, 'r') as f:
    demos = json.load(f)

for demo_key in tqdm(demos):
    traj = 'Instruction:'+demos[demo_key].split("Instruction:")[-1]
    traj = traj.split('Your score (min 0.0, max 1.0)')[0]

    input_ids1 = tokenizer(traj, return_tensors="pt")["input_ids"]
    reward_pred = model(input_ids1)
    reward_pred = reward_pred.detach().cpu()

    results.append({
        'traj' : traj,
        'pred_reward' : float(reward_pred)
    })
    
    # save 
    with open('/home/seungyoun/LLM2Act/demo_files/selfplay_dense_reward_pred.json', 'w') as f:
        json.dump(results, f)

print(f'Done.')

