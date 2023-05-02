from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch 
from typing import Optional, Dict, Sequence
import transformers

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

class RewardModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        print(f'Loading model ...') 
        self.llm = model # AutoCausalLM
        self.linear = nn.Linear(4096, 1)

        self.freeze_layers(30)

        # print trainable parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Number of trainable parameters: {num_params / 1_000_000:.2f}M')

    def freeze_layers(self, n_layers_to_freeze):
        n_layers = 32  # Total number of layers
        for name, param in self.llm.named_parameters():
            # If the parameter belongs to one of the first n_layers_to_freeze layers, set requires_grad to False
            if "layers" in name:
                layer_idx = int(name.split(".")[2])
                if layer_idx < n_layers_to_freeze:
                    param.requires_grad = False

    def forward(self, input_ids, return_loss=True):
        llm_out = self.llm(input_ids, output_hidden_states=True)
        # print(len(llm_out.hidden_states)) 33
        last_hidden_state = llm_out.hidden_states[-1]
        
        pooler_out = last_hidden_state.mean(dim=1) # (batch_size, n_embd)
        score = self.linear(pooler_out) # (batch_size, 1)
        # sigmoid out 
        score = torch.sigmoid(score)
        #print(score)

        return score

        #return score.squeeze()

if __name__=="__main__":
    tokenizer = LlamaTokenizer.from_pretrained("/home/seungyoun/stanford_alpaca/ckpt/7B/tokenizer")
    model = LlamaForCausalLM.from_pretrained("/home/seungyoun/stanford_alpaca/ckpt/7B/llama-7b")

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

    reward_model = RewardModel(model)
    
    # load json file
    path = '/home/seungyoun/webshop/demo_files/webshop_demos.json'
    import json
    with open(path) as f:
        data = json.load(f) # list of dict (traj,score)

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
