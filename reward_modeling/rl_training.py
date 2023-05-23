from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM,AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
import torch.nn as nn
import torch 
from typing import Optional, Dict, Sequence
import transformers
from reward_model import RewardModel

from dataclasses import dataclass, field
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

if __name__=="__main__":
    parser = transformers.HfArgumentParser((TrainingArguments))
    training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained("/home/seungyoun/stanford_alpaca/ckpt/7B/tokenizer")
    model = AutoModelForCausalLMWithValueHead.from_pretrained("/home/seungyoun/stanford_alpaca/ckpt/webshop_full_onlysucess_chat_full/checkpoint-264")

    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )

    # initialize trainer
    ppo_config = PPOConfig(
        batch_size=1,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
    )
    
    # load json file
    path = '/home/seungyoun/LLM2Act/demo_files/selfplay_dense_reward_pred.json'
    import json
    with open(path) as f:
        data = json.load(f) # list of dict (traj,pred_reward)
    print(data[0].keys())

    # create a ppo trainer
    ppo_trainer = PPOTrainer(ppo_config, model, None, tokenizer)

    for data_ in data:
        # process data
        traj = data_['traj']
        reward = torch.tensor(data_['pred_reward'])
        instruction_part = traj.split('Action:')[0].strip()
        rest_part = traj[len(instruction_part):].strip()

        # tokenize
        query_tensor = tokenizer.encode(instruction_part, return_tensors="pt")
        response_tensor = tokenizer.encode(rest_part, return_tensors="pt")
        print(query_tensor[0])
        print('=======')
        print(rest_part)
        print(reward)

        train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], [reward])
        exit()
    # criterion
    criterion = nn.MSELoss()

    # optimizer
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=2e-5)

    # train one epoch
    for idx,batch in enumerate(data):
        traj = batch['traj']
        reward = torch.tensor(batch['score']).unsqueeze(0)

        input_ids = tokenizer(traj, return_tensors="pt")["input_ids"]
        reward_pred = reward_model(input_ids)
        
        loss = criterion(reward_pred, reward)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print log
        if idx % 10 == 0:
            print(f'[{idx}/{len(data)}] loss: {loss.item():.4f}')
    
    # save model 
    torch.save(reward_model.state_dict(), 'ckpt/reward_model.pt')
