import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers.models.fnet import FNetConfig, FNetModel, FNetTokenizer
import config


def load_fnet(args):
    config = FNetConfig.from_pretrained(args.model_dir)
    model = FNetModel.from_pretrained(args.model_dir, config=config)
    
    return model


class CoLAModel(nn.Module):
    def __init__(self, args):
        super(CoLAModel, self).__init__()
        self.args = args
        self.fnet = load_fnet(self.args)
        self.fc = nn.Linear(1024, 2, bias=False)
        
    def forward(self, input_ids, token_type_ids, labels):
        outputs = self.fnet(input_ids, token_type_ids=token_type_ids)
        logits = self.fc(outputs[0])
        
        indices = torch.tensor([0]).to(self.args.device)
        logits = torch.squeeze(torch.index_select(logits, 1, indices),dim=1)
        
        return logits
