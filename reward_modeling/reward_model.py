import torch 
import torch.nn as nn 

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