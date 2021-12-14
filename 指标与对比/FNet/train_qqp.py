import datetime
import random
import time
from functools import partial
import logging
import numpy as np
import paddle
import paddle.nn as nn
import utils
from paddle.optimizer import AdamW
from paddlenlp.data import Dict, Pad, Stack
from paddlenlp.datasets import load_dataset
from sklearn.metrics import accuracy_score

from tokenizer import FNetTokenizer
from modeling import FNetForSequenceClassification
from paddle_metric import F1_score




def evaluate(model, criterion, data_loader, metric, print_freq=100, logger=logger):
    model.eval()
    metric.reset()
    metric_logger = utils.MetricLogger(logger=logger, delimiter="  ")
    header = "Test:"
    labels, preds = [], []
    with paddle.no_grad():
        for batch in metric_logger.log_every(data_loader, print_freq, header):
            inputs = {"input_ids": batch[0], "token_type_ids": batch[1]}
            label = batch[2]
            logits = model(**inputs)
            loss = criterion(
                logits.reshape([-1, model.num_classes]),
                label.reshape([-1, ]), )
            metric_logger.update(loss=loss.item())
            corrects = metric.compute(logits, label)
            metric.update(corrects=corrects, labels=label)
            labels.extend(label.cpu().numpy())
            preds.extend(logits.cpu().numpy().argmax(-1))
        f1_global_avg = metric.accumulate()
        acc_global_avg = accuracy_score(labels, preds)
        score_global_avg = (f1_global_avg + acc_global_avg) / 2
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(" * Mean(F1+ACC) {score_global_avg:.10f}".format(
        score_global_avg=score_global_avg))
    logger.info(" * Mean(F1+ACC) {score_global_avg:.10f}".format(
        score_global_avg=score_global_avg))
    return score_global_avg


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def convert_example(example, tokenizer, max_length=128):
    labels = np.array([example["labels"]], dtype="int64")
    text = (example["sentence1"], example["sentence2"])
    example = tokenizer(*text, max_seq_len=max_length)
    return {
        "input_ids": example["input_ids"],
        "token_type_ids": example["token_type_ids"],
        "labels": labels,
    }


def load_data(args, tokenizer):
    print("Loading data")
    train_ds = load_dataset("glue", args.task_name, splits="train")
    validation_ds = load_dataset("glue", args.task_name, splits="dev")
    
    trans_func = partial(
        convert_example, tokenizer=tokenizer, max_length=args.max_length)
    train_ds = train_ds.map(trans_func, lazy=False)
    validation_ds = validation_ds.map(trans_func, lazy=False)
    
    train_sampler = paddle.io.BatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=False)
    validation_sampler = paddle.io.BatchSampler(
        validation_ds, batch_size=args.batch_size, shuffle=False)
    
    return train_ds, validation_ds, train_sampler, validation_sampler


def main(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(args.logging_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if args.output_dir:
        utils.mkdir(args.output_dir)
    print(args)
    scaler = None
    if args.fp16:
        scaler = paddle.amp.GradScaler()
    paddle.set_device(args.device)
    
    if args.seed is not None:
        set_seed(args.seed)
    
    tokenizer = FNetTokenizer.from_pretrained(args.model_name_or_path)
    batchify_fn = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "labels": Stack(dtype="int64"), }): fn(samples)
    train_dataset, validation_dataset, train_sampler, validation_sampler = load_data(
        args, tokenizer)
    
    train_data_loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=batchify_fn, )
    validation_data_loader = paddle.io.DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        num_workers=args.workers,
        collate_fn=batchify_fn, )
    
    print("Creating model")
    model = FNetForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_classes=2)
    classifier_weights = paddle.load(
        "./classifier_weights/paddle_classifier_weights_large.bin")
    model.load_dict(classifier_weights)
    
    print("Creating criterion")
    criterion = nn.CrossEntropyLoss()
    
    print("Creating lr_scheduler")
    lr_scheduler = utils.get_scheduler(
        learning_rate=args.lr,
        scheduler_type=args.lr_scheduler_type,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * len(train_data_loader), )
    
    print("Creating optimizer")
    # Split weights in two groups, one with weight decay and the other not.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        epsilon=1e-6,
        apply_decay_param_fun=lambda x: x in decay_params, )
    metric = F1_score()
    
    if args.test_only:
        evaluate(model, criterion, validation_data_loader, metric)
        return
    
    print("Start training")
    start_time = time.time()
    steps = last_improvement = 0
    best_score = 0.0
    for epoch in range(args.num_train_epochs):
        model.train()
        metric_logger = utils.MetricLogger(logger=logger, delimiter="  ")
        metric_logger.add_meter(
            "lr", utils.SmoothedValue(
                window_size=1, fmt="{value}"))
        metric_logger.add_meter(
            "sentence/s", utils.SmoothedValue(
                window_size=10, fmt="{value}"))
        
        header = "Epoch: [{}]".format(epoch)
        for batch in metric_logger.log_every(train_data_loader, args.print_freq, header):
            model.train()
            inputs = {"input_ids": batch[0], "token_type_ids": batch[1]}
            labels = batch[2]
            start_time1 = time.time()
            with paddle.amp.auto_cast(
                    enable=scaler is not None,
                    custom_white_list=["layer_norm", "softmax", "gelu"], ):
                logits = model(**inputs)
                loss = criterion(
                    logits.reshape([-1, model.num_classes]),
                    labels.reshape([-1, ]), )
            
            optimizer.clear_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            steps += 1
            
            if steps % args.eval_freq == 0:
                score = evaluate(model, criterion, validation_data_loader, metric, logger=logger)
                if score > best_score:
                    best_score = score
                    last_improvement = steps
                elif steps - last_improvement > args.early_stop:
                    logger.info("It's been a long time since the last improvement! Early stop!")
                    total_time = time.time() - start_time
                    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                    print("Training time {}".format(total_time_str))
                    logger.info("Training time {}".format(total_time_str))
                    logger.info("* Mean(F1+ACC) : {:8f} *".format(score))
                    return best_score
            
            batch_size = inputs["input_ids"].shape[0]
            metric_logger.update(loss=loss.item(), lr=lr_scheduler.get_lr())
            metric_logger.meters["sentence/s"].update(batch_size /
                                                      (time.time() - start_time1))
        
        if args.output_dir:
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
    
    score = evaluate(model, criterion, validation_data_loader, metric, logger=logger)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Total time {}".format(total_time_str))
    logger.info("Total time {}".format(total_time_str))
    logger.info("* Mean(F1+ACC) : {:8f} *".format(score))
    return score


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description="Paddle QQP Classification Training", add_help=add_help)
    parser.add_argument("--task_name", default="qqp",
                        help="the name of the glue task to train on.")
    parser.add_argument("--logger_file", default="sst2_log_base.txt",
                        help="path to save logging information")
    parser.add_argument("--model_name_or_path", default="/root/autodl-tmp/PaddleFNet/model/paddle/fnet-large/",
                        help="path to pretrained model or model identifier from huggingface.co/models.", )
    parser.add_argument("--device", default="gpu", help="device")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_length", type=int, default=128,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,")
    parser.add_argument("--num_train_epochs", default=2, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--workers", default=0, type=int,
                        help="number of data loading workers (default: 16)", )
    parser.add_argument("--lr", default=3e-5, type=float, help="initial learning rate")
    parser.add_argument("--weight_decay", default=1e-2, type=float,
                        help="weight decay (default: 1e-2)",
                        dest="weight_decay", )
    parser.add_argument("--lr_scheduler_type", default="linear",
                        help="the scheduler type to use.",
                        choices=["linear", "cosine", "polynomial"], )
    parser.add_argument("--num_warmup_steps", default=3000, type=int,
                        help="number of steps for the warmup in the lr scheduler.", )
    parser.add_argument("--print_freq", default=100, type=int, help="print frequency")
    parser.add_argument("--eval_freq", default=2000, type=int, help="evaluation frequency")
    parser.add_argument("--early_stop", default=10000, type=int, help="early stop iters")
    parser.add_argument("--output_dir", default="outputs", help="path where to save")
    parser.add_argument("--test_only", help="only test the model", action="store_true", )
    parser.add_argument("--seed", default=1234, type=int,
                        help="a seed for reproducible training.")
    # Mixed precision training parameters
    parser.add_argument("--fp16", action="store_true",
                        help="whether or not mixed precision training")
    
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    score = main(args)
    print("* Mean(F1+ACC) : {:8f} *".format(score))
