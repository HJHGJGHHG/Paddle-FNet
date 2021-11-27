import torch
import random
import numpy as np
import torch.fft as fft
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers.models.fnet import FNetConfig, FNetModel, FNetTokenizer
import config
from model import CoLAModel
from Train_Test import train, eval_model


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


class CoLADataset(Dataset):
    def __init__(self, args, dataset, phase):
        super().__init__()
        self.args = args
        self.instances = dataset[phase]
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        return self.convert_to_tensors(self.instances[idx])
    
    def convert_to_tensors(self, instance):
        sentence = instance['sentence']
        label = instance['label']
        
        input_ids = self.args.tokenizer.encode_plus(sentence, padding='max_length', max_length=256)['input_ids']
        token_type_ids = self.args.tokenizer.encode_plus(sentence, padding='max_length', max_length=256)[
            'token_type_ids']
        return torch.LongTensor(input_ids).to(self.args.device), \
               torch.LongTensor(token_type_ids).to(self.args.device), \
               torch.LongTensor([label]).to(self.args.device)


def main():
    args = config.args_initialization()
    dataset = load_dataset('glue', 'cola')
    args.tokenizer = FNetTokenizer.from_pretrained(args.model_dir)
    set_seed(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_dataset = CoLADataset(args, dataset, phase='train')
    dev_dataset = CoLADataset(args, dataset, phase='validation')
    # test_dataset = CoLADataset(args, dataset, phase='test')
    
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size)
    dev_iter = DataLoader(dev_dataset, batch_size=args.batch_size)
    # test_iter = DataLoader(test_dataset, batch_size=args.batch_size)
    
    model = CoLAModel(args).to(args.device)
    
    train(args, model, train_iter, dev_iter)
    eval_model(args, model, dev_iter)
    
    # torch.save(model.state_dict, '/root/autodl-tmp/FNet/models/model.pth')
    
    return model


if __name__ == '__main__':
    main()