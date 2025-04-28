import torch.nn as nn

class BaseOutputHead(nn.Module):
    def forward(self, x):
        raise NotImplementedError("Each head must implement the forward method.")
    
    def get_loss_fn(self):
        raise NotImplementedError("Each head must provide its corresponding loss function.")
