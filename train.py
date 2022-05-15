import time
import torch
import torch.nn.functional as F
import argparse
from hps import get_hyper_param
from model.gnn_bc import GNNBC
from util import load_dataset, root, get_mask, get_accuracy, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
args = parser.parse_args()
set_seed(0xC0FFEE)
epochs = 1000
patience = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = root + "/checkpoint"
feat, label, n, nfeat, nclass, adj = load_dataset(args.dataset, norm=True, device=device)
hp = get_hyper_param(args.dataset)


def train(model, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    result = model(feat=feat, adj=adj)
    loss = F.nll_loss(result[train_mask], label[train_mask])
    loss.backward()
    optimizer.step()
    return get_accuracy(result[train_mask], label[train_mask]), loss.item()


def test(model, test_mask):
    model.eval()
    with torch.no_grad():
        result = model(feat=feat, adj=adj)
        loss = F.nll_loss(result[test_mask], label[test_mask].to(device))
        return get_accuracy(result[test_mask], label[test_mask]), loss.item()


def validate(model, val_mask) -> float:
    model.eval()
    with torch.no_grad():
        result = model(feat=feat, adj=adj)
        return get_accuracy(result[val_mask], label[val_mask])


def run():
    train_mask, test_mask, val_mask = get_mask(label, 0.6, 0.2, device=device)
    model = GNNBC(
        n=n,
        nclass=nclass,
        nfeat=nfeat,
        nlayer=hp["layer"],
        lambda_1=hp["lambda_1"],
        lambda_2=hp["lambda_2"],
        dropout=hp["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {'params': model.params1, 'weight_decay': hp["wd1"]},
            {'params': model.params2, 'weight_decay': hp["wd2"]}
        ],
        lr=hp["lr"]
    )
    checkpoint_file = "{}/{}-{}.pt".format(checkpoint_path, model.__class__.__name__, args.dataset)
    tolerate = 0
    best_loss = 100
    for epoch in range(epochs):
        if tolerate >= patience:
            break
        train_acc, train_loss = train(model, optimizer, train_mask)
        test_acc, test_loss = test(model, test_mask)
        if train_loss < best_loss:
            tolerate = 0
            best_loss = train_loss
        else:
            tolerate += 1
        message = "Epoch={:<4} | Tolerate={:<3} | Train_acc={:.4f} | Train_loss={:.4f} | Test_acc={:.4f} | Test_loss={:.4f}".format(
            epoch,
            tolerate,
            train_acc,
            train_loss,
            test_acc,
            test_loss
        )
        print(message)
    val_acc = validate(model, val_mask)
    torch.save(model.state_dict(), checkpoint_file)
    print("Validate accuracy {:.4f}.".format(val_acc))
    return val_acc


if __name__ == '__main__':
    run()
