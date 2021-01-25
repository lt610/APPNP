import argparse
import torch
import torch.nn.functional as F
from networks import Net
from torch import tensor
from torch.optim import Adam
from utils import load_data, random_splits
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='dblp')
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=20)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.1)
args = parser.parse_args()

dataset = load_data(args.dataset)
args.num_features = dataset.x.shape[1]
args.num_classes = len(set(np.array(dataset.y)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(args).to(device)
all_acc = []

for _ in range(args.runs):

    data = random_splits(dataset, args.num_classes)
    data = data.to(device)

    if args.normalize_features:
        data.x = F.normalize(data.x, p=1)

    model.reset_parameters()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float('inf')
    val_loss_history = []

    for epoch in range(args.epochs):

        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        pred = model(data.x, data.edge_index)
        val_loss = F.nll_loss(pred[data.val_mask], data.y[data.val_mask]).item()
        pred = model(data.x, data.edge_index).max(1)[1]
        acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_acc = acc

        #print(epoch, acc)

        val_loss_history.append(val_loss)
        if args.early_stopping > 0 and epoch > args.epochs // 2:
            tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
            if val_loss > tmp.mean().item():
                break

    print(test_acc)
    all_acc.append(test_acc)

print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))


