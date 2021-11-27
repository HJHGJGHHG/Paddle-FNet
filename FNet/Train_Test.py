import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from transformers import AdamW, get_linear_schedule_with_warmup


def train(args, model, train_iter, val_iter):
    # optimizer
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    #optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-6
    )
    
    # scheduler
    num_training_steps = args.num_epochs * len(train_iter) / args.batch_size
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=num_training_steps)
    # loss function
    #loss_func = nn.BCEWithLogitsLoss()
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(args.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, args.num_epochs))
        model.train()
        losses, corrects = [], []
        for sample in tqdm(train_iter):
            model.train()
            optimizer.zero_grad()
            outputs = model(*sample)
            
            target = torch.squeeze(sample[2],dim=1).to(args.device)
            loss = loss_func(outputs, target)
            losses.append(loss.item())
            pred = (torch.max(outputs, 1)[1].view(target.size()).data == target.data).sum().item()
            corrects.append(pred)
            
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # 一个Epoch训练完毕，输出train_loss
        print('Epoch: {0}   Average Train Loss: {1:>5.8}    Average Train ACC: {2:>5.8}'.format(epoch + 1, np.mean(losses), np.sum(corrects) / len(train_iter) /args.batch_size))
    # 训练结束


def eval_model(args, model, val_iter):
    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        model.eval()
        test_loss, test_correct = [], []
        for sample in tqdm(val_iter):
            pred = []
            outputs = model(*sample)
            
            target = torch.squeeze(sample[2],dim=1).to(args.device)
            pred = (torch.max(outputs, 1)[1].view(target.size()).data == target.data).sum().item()
            loss = loss_func(outputs, target)
            test_loss.append(loss.item())
            test_correct.append(pred)
        print('Average Val Loss: {0:>5.8}  Average ACC: {1:5.8}'.format(np.mean(test_loss),
                                                                        np.sum(test_correct) / len(val_iter)/args.batch_size))