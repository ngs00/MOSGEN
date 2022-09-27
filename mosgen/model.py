import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool


class MOSGEN(torch.nn.Module):
    def __init__(self, dim_node_feats, dim_edge_feats, dim_out):
        super(MOSGEN, self).__init__()
        self.gc1 = TransformerConv(in_channels=dim_node_feats, out_channels=64, edge_dim=dim_edge_feats)
        self.gn1 = LayerNorm(64)
        self.gc2 = TransformerConv(in_channels=64, out_channels=16, edge_dim=dim_edge_feats)
        self.gn2 = LayerNorm(16)
        self.fc = torch.nn.Linear(16, dim_out)

    def forward(self, g):
        h = F.relu(self.gn1(self.gc1(g.x, g.edge_index, g.edge_attr)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index, g.edge_attr)))
        z = global_mean_pool(h, g.batch)
        out = self.fc(z)

        return out

    def fit(self, dataset, optimizer, criterion):
        self.train()
        train_loss = 0

        for i in range(0, 10):
            batch = dataset.get_batch(n_per_class=3)
            batch.cuda()

            preds = self(batch)
            loss = criterion(preds, batch.y.flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss

    def test(self, dataset):
        self.eval()

        loader = DataLoader(dataset.dataset, batch_size=64)
        list_preds = list()
        list_targets = list()

        with torch.no_grad():
            for batch in loader:
                batch.cuda()

                preds = self(batch)
                list_preds.append(preds)
                list_targets.append(batch.y)

            preds = torch.vstack(list_preds)
            targets = torch.vstack(list_targets).flatten()

        _, pred_labels = torch.max(preds, 1)
        correct = (pred_labels == targets).sum().item()

        return pred_labels, 100 * (correct / preds.shape[0]), preds
