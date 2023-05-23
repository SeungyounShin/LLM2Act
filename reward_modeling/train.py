#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from accelerate import Accelerator
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from transformers import Trainer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

from tqdm import tqdm 

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        #del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class RLDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(RLDataset, self).__init__()
        logging.warning("Loading data...")
        
        #data_path = '/home/seungyoun/LLM2Act/demo_files/selfplay_dense_reward_pred.json'
        import json
        with open(data_path) as f:
            data = json.load(f) # list of dict (traj,pred_reward)


        to_list = list()
        for data_ in data:
            # process data
            traj = data_['traj']
            reward = torch.tensor(data_['pred_reward'])
            instruction_part = traj.split('Action:')[0].strip()
            rest_part = traj[len(instruction_part):].strip()
            if len(rest_part) > 2000:
                rest_part = rest_part[:2000]

            # tokenize
            query_tensor = tokenizer.encode(instruction_part, return_tensors="pt")[0]
            response_tensor = tokenizer.encode(rest_part, return_tensors="pt")[0]

            to_list.append({
                'query_tensor' : query_tensor,
                'response_tensor' : response_tensor,
                'reward' : reward,
            })
        
        self.data = to_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(query_tensor=self.data[i]['query_tensor'],
                    response_tensor=self.data[i]['response_tensor'],
                    reward=self.data[i]['reward'])


@dataclass
class DataCollatorForRLDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        query_tensor, response_tensor,reward = tuple( [instance[key] for instance in instances] for key in ("query_tensor", "response_tensor","reward"))
        query_tensor = torch.nn.utils.rnn.pad_sequence(
            query_tensor, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        response_tensor = torch.nn.utils.rnn.pad_sequence(
            response_tensor, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            query_tensor=query_tensor,
            response_tensor = response_tensor,
            reward=reward,
        )


def make_rl_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = RLDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForRLDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=DataCollatorForRLDataset)

def save_fsdp_model(model, save_path):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import (
        FullyShardedDataParallel,
        CPUOffload,
        )
    from torch.distributed.fsdp import StateDictType,ShardedStateDictConfig
    
    #model = self.accelerator.unwrap_model(self.model).save_pretrained(save_directory)

    FSDP.set_state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        state_dict_config = ShardedStateDictConfig(offload_to_cpu=True))

    param_state_dict = model.state_dict()
    torch.save(param_state_dict, save_path)
    print(f'model saved on ==> {save_path}')


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    '''model = transformers.AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,cache_dir=training_args.cache_dir,)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )'''

    #tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)
    #model = LlamaForCausalLM.from_pretrained(model_args.model_name_or_path)

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

    data_module = make_rl_data_module(tokenizer=tokenizer, data_args=data_args)

    # initialize trainer
    ppo_config = PPOConfig(
        batch_size=1,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,

    )

    ppo_trainer = PPOTrainer(ppo_config, model, None, tokenizer, 
                             dataset = data_module['train_dataset'],
                             data_collator=data_module['data_collator'])

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query = batch.tokenizer[0]['query_tensor']
        responed = batch.tokenizer[0]['response_tensor']
        reward = batch.tokenizer[0]['reward']

        stats = ppo_trainer.step([query], [responed], [reward])
        
        
        print(epoch)
        if epoch%10==0:
            ppo_trainer._save_pretrained(training_args.output_dir)
            #save_fsdp_model(model=ppo_trainer.model, save_path=training_args.output_dir)
            print(f'saved {epoch}')

    #exit()
    #ppo_trainer.save_pretrained(training_args.output_dir + f"step_{epoch}")
    #print(f"saved")


    '''trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)'''


if __name__ == "__main__":
    train()
