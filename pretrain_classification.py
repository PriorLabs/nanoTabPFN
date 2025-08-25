import argparse
import torch
import numpy as np

from torch import nn
from functools import partial

from nanotabpfn.callbacks import ConsoleLoggerCallback
from nanotabpfn.priors import PriorDumpDataLoader
from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.train import train
from nanotabpfn.utils import get_default_device, set_randomness_seed
from nanotabpfn.interface import NanoTabPFNClassifier

from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("-priordump", type=str, default="/50x3_3_100k_classification.h5", help="path to the prior dump")
parser.add_argument("-saveweights", type=str, default="nanotabpfn_weights.pth", help="path to save the trained model to")
parser.add_argument("-heads", type=int, default=6, help="number of attention heads")
parser.add_argument("-embeddingsize", type=int, default=192, help="the size of the embeddings used for the cells")
parser.add_argument("-hiddensize", type=int, default=768, help="size of the hidden layer of the mlps")
parser.add_argument("-layers", type=int, default=6, help="number of transformer layers")
parser.add_argument("-batchsize", type=int, default=1, help="batch size used during training (before gradient accumulation)")
parser.add_argument("-accumulate", type=int, default=1, help="number of gradients to accumulate before updating the weights")
parser.add_argument("-lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("-steps", type=int, default=100, help="number of steps that constitute one epoch (important for lr scheduler)")
parser.add_argument("-epochs", type=int, default=10000, help="number of epochs to train for")
parser.add_argument("-loadcheckpoint", type=str, default=None, help="checkpoint from which to continue training")

args = parser.parse_args()

set_randomness_seed(2402)

device = get_default_device()
ckpt = None
if args.loadcheckpoint:
    ckpt = torch.load(args.loadcheckpoint)

prior = PriorDumpDataLoader(filename=args.priordump, num_steps=args.steps, batch_size=args.batchsize, device=device, starting_index=args.steps*(ckpt['epoch'] if ckpt else 0))


criterion = nn.CrossEntropyLoss()

model = NanoTabPFNModel(
    num_attention_heads=args.heads,
    embedding_size=args.embeddingsize,
    mlp_hidden_size=args.hiddensize,
    num_layers=args.layers,
    num_outputs=prior.max_num_classes,
)

if ckpt:
    model.load_state_dict(ckpt['model'])

datasets = []
datasets.append(train_test_split(*load_iris(return_X_y=True), test_size=0.5, random_state=42))
datasets.append(train_test_split(*load_wine(return_X_y=True), test_size=0.5, random_state=42))
datasets.append(train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.5, random_state=42))


class EvaluationLoggerCallback(ConsoleLoggerCallback):
    def __init__(self, datasets):
        self.datasets = datasets

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        classifier = NanoTabPFNClassifier(model, device)
        scores = []
        for X_train, X_test, y_train, y_test in self.datasets:
            classifier.fit(X_train, y_train)
            pred = classifier.predict(X_test)
            scores.append(accuracy_score(y_test, pred))
        avg_score = sum(scores) / len(scores)
        print(f'epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg accuracy {avg_score:.3f}',
              flush=True)


callbacks = [EvaluationLoggerCallback(datasets)]

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=criterion,
    epochs=args.epochs,
    accumulate_gradients=args.accumulate,
    lr=args.lr,
    device=device,
    callbacks=callbacks,
    ckpt=ckpt
)

torch.save(trained_model.to('cpu').state_dict(), args.saveweights)
