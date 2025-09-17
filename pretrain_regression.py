import argparse

import torch
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.metrics import r2_score

from nanotabpfn.callbacks import ConsoleLoggerCallback
from nanotabpfn.evaluation import get_openml_predictions, TOY_TASKS_REGRESSION
from nanotabpfn.interface import NanoTabPFNRegressor
from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.priors import PriorDumpDataLoader
from nanotabpfn.train import train
from nanotabpfn.utils import get_default_device, set_randomness_seed, make_global_bucket_edges

parser = argparse.ArgumentParser()

parser.add_argument("-priordump", type=str, default="/50x3_1280k_regression.h5", help="path to the prior dump")
parser.add_argument("-saveweights", type=str, default="nanotabpfn_weights.pth", help="path to save the trained model to")
parser.add_argument("-savebuckets", type=str, default="nanotabpfn_buckets.pth", help="path to save the bucket edges to")
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
parser.add_argument("-n_buckets", type=int, default=100, help="number of buckets for the data loader")

args = parser.parse_args()

set_randomness_seed(2402)

device = get_default_device()
ckpt = None
if args.loadcheckpoint:
    ckpt = torch.load(args.loadcheckpoint)

prior = PriorDumpDataLoader(filename=args.priordump, num_steps=args.steps, batch_size=args.batchsize, device=device, starting_index=args.steps*(ckpt['epoch'] if ckpt else 0))

model = NanoTabPFNModel(
    num_attention_heads=args.heads,
    embedding_size=args.embeddingsize,
    mlp_hidden_size=args.hiddensize,
    num_layers=args.layers,
    num_outputs=args.n_buckets,
)

bucket_edges = make_global_bucket_edges(
    filename=args.priordump,
    n_buckets=args.n_buckets,
    device=device,
)

torch.save(
    bucket_edges,
    args.savebuckets,
)

if ckpt:
    model.load_state_dict(ckpt['model'])

dist = FullSupportBarDistribution(bucket_edges)

class EvaluationLoggerCallback(ConsoleLoggerCallback):
    def __init__(self, tasks):
        self.tasks = tasks

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        regressor = NanoTabPFNRegressor(model, dist, device)
        predictions = get_openml_predictions(model=regressor, tasks=self.tasks)
        scores = []
        for dataset_name, (y_true, y_pred, _) in predictions.items():
            scores.append(r2_score(y_true, y_pred))
        avg_score = sum(scores) / len(scores)
        print(f'epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg r2 score {avg_score:.3f}',
              flush=True)


callbacks = [EvaluationLoggerCallback(TOY_TASKS_REGRESSION)]

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=dist,
    epochs=args.epochs,
    accumulate_gradients=args.accumulate,
    lr=args.lr,
    device=device,
    callbacks=callbacks,
    ckpt=ckpt
)

torch.save(trained_model.to('cpu').state_dict(), args.saveweights)
