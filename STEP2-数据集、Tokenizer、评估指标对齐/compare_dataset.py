import torch
import argparse
import torch.fft as fft
import paddle
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from reprod_log import ReprodDiffHelper, ReprodLogger

from transformers.models.fnet import FNetForSequenceClassification as PTFetForSequenceClassification
from modeling import FNetForSequenceClassification as PDFNetForSequenceClassification
from transformers.models.fnet import FNetTokenizer as PTFNetTokenizer
from tokenizer import FNetTokenizer as PDFNetTokenizer


class CoLADataset_torch(torch.utils.data.Dataset):
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
        
        input_ids = self.args.torch_tokenizer.encode_plus(sentence, padding='max_length', max_length=256)['input_ids']
        token_type_ids = self.args.torch_tokenizer.encode_plus(sentence, padding='max_length', max_length=256)[
            'token_type_ids']
        return torch.LongTensor(input_ids).to(self.args.device), \
               torch.LongTensor(token_type_ids).to(self.args.device), \
               torch.LongTensor([label]).to(self.args.device)


class CoLADataset_paddle(paddle.io.Dataset):
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
        
        input_ids = self.args.paddle_tokenizer.encode(sentence, pad_to_max_seq_len=True, max_seq_len=256)['input_ids']
        token_type_ids = self.args.paddle_tokenizer.encode(sentence, pad_to_max_seq_len=True, max_seq_len=256)[
            'token_type_ids']
        
        return paddle.to_tensor(input_ids, dtype='int64'), \
               paddle.to_tensor(token_type_ids, dtype='int64'), \
               paddle.to_tensor([label], dtype='int64')

    
def compare_dataset(torch_dataset, paddle_dataset, torch_iter, paddle_iter):
    diff_helper = ReprodDiffHelper()
    logger_paddle_data = ReprodLogger()
    logger_torch_data = ReprodLogger()

    logger_paddle_data.add("length", np.array(len(paddle_dataset)))
    logger_torch_data.add("length", np.array(len(torch_dataset)))
    for idx, (paddle_batch, torch_batch) in enumerate(zip(paddle_iter, torch_iter)):
        if idx >= 5:
            break
        for i, k in enumerate([0, 1, 2]):  # "input_ids", "token_type_ids", "labels"
            logger_paddle_data.add(f"dataloader_{idx}_{k}",
                                   paddle_batch[i].numpy())
            logger_torch_data.add(f"dataloader_{idx}_{k}",
                                  torch_batch[k].cpu().numpy())

    diff_helper.compare_info(logger_paddle_data.data, logger_torch_data.data)
    diff_helper.report(path="data_diff.log")
            
    
def main():
    parser = argparse.ArgumentParser(description="FNet-CoLA")

    parser.add_argument("--seed", type=int, default=1234, help='随机种子')
    parser.add_argument("--torch-dir", type=str, default='google/fnet-large', help='模型位置')
    parser.add_argument("--paddle-dir", type=str, default='../model/paddle/fnet-large', help='模型位置')
    parser.add_argument("--batch-size", type=int, default=4, help='Batch Size')
    parser.add_argument("--lr", type=float, default=1e-5, help='Learning Rate')
    parser.add_argument("--warmup", type=int, default=0, help='Warmup Steps')
    parser.add_argument("--num-epochs", type=int, default=3, help='Epoch 数')
    args = parser.parse_args(args=[])
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_dataset('glue', 'cola')
    args.paddle_tokenizer = PDFNetTokenizer.from_pretrained(args.paddle_dir)
    args.torch_tokenizer = PTFNetTokenizer.from_pretrained(args.torch_dir)
    
    # dev dataset
    torch_dev_dataset = CoLADataset_torch(args, dataset, phase='validation')
    paddle_dev_dataset = CoLADataset_paddle(args, dataset, phase='validation')
    # dev dataset
    torch_train_dataset = CoLADataset_torch(args, dataset, phase='train')
    paddle_train_dataset = CoLADataset_paddle(args, dataset, phase='train')
    
    # dev iter
    paddle_dev_iter = paddle.io.DataLoader(paddle_dev_dataset, batch_size=args.batch_size)
    torch_dev_iter = torch.utils.data.DataLoader(torch_dev_dataset, batch_size=args.batch_size)
    # train iter
    paddle_train_iter = paddle.io.DataLoader(paddle_train_dataset, batch_size=args.batch_size)
    torch_train_iter = torch.utils.data.DataLoader(torch_train_dataset, batch_size=args.batch_size)
    
    return args, torch_dev_dataset, paddle_dev_dataset, torch_dev_iter, paddle_dev_iter


if __name__ == '__main__':
    main()
