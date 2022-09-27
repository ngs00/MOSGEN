import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import f1_score


def binary_acc(targets, preds):
    preds = torch.round(preds)

    correct = (preds == targets).sum().float()
    acc = 100 * (correct / preds.shape[0])

    return acc.item()


def f1_acc(targets, preds):
    return f1_score(targets.cpu().numpy(), torch.round(preds).cpu().numpy())


class MOSGEN(torch.nn.Module):
    def __init__(self, dim_node_feats, dim_edge_feats):
        super(MOSGEN, self).__init__()
        self.gc1 = TransformerConv(in_channels=dim_node_feats, out_channels=64, edge_dim=dim_edge_feats)
        self.gc2 = TransformerConv(in_channels=64, out_channels=16, edge_dim=dim_edge_feats)
        self.fc = torch.nn.Linear(16, 1)

    def forward(self, g):
        h = F.relu(self.gc1(g.x, g.edge_index, g.edge_attr))
        h = F.relu(self.gc2(h, g.edge_index, g.edge_attr))
        z = global_mean_pool(h, g.batch)
        out = torch.sigmoid(self.fc(z))

        return out

    def fit(self, dataset, optimizer, criterion, n_per_class=16):
        self.train()

        train_loss = 0
        train_acc = 0

        for i in range(0, 10):
            batch = dataset.get_batch(n_per_class=n_per_class)
            batch.cuda()

            preds = self(batch)
            loss = criterion(preds, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()
            train_acc += binary_acc(batch.y, preds)

        return train_loss / 10, train_acc / 10

    def test(self, dataset):
        self.eval()

        loader = DataLoader(dataset.dataset, batch_size=64)
        list_preds = list()
        list_targets = list()

        with torch.no_grad():
            for batch in loader:
                batch.cuda()

                preds = self(batch)
                list_preds.append(torch.round(preds))
                list_targets.append(batch.y)

            preds = torch.vstack(list_preds)
            targets = torch.vstack(list_targets)

        acc = binary_acc(targets, preds)
        f1 = f1_acc(targets, preds)

        return preds, acc, f1
