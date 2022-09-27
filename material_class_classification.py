import numpy
import torch
from itertools import chain
from util.data import load_dataset
from mosgen.model import MOSGEN


n_folds = 3
n_epochs = 5000


dataset = load_dataset('datasets/ir_dataset.json')
kfolds = dataset.get_k_fold(k=n_folds, random_seed=0)
accs = list()
targets = list()
preds = list()


for k in range(0, n_folds):
    dataset_train = kfolds[k][0]
    dataset_test = kfolds[k][1]
    model = MOSGEN(dim_node_feats=dataset.dim_node_feats,
                   dim_edge_feats=dataset.dim_edge_feats,
                   dim_out=dataset.n_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(0, n_epochs):
        loss_train = model.fit(dataset_train, optimizer, criterion)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, n_epochs, loss_train))

    labels_pred, acc_test, _ = model.test(dataset_test)
    accs.append(acc_test)
    targets.append(kfolds[k][1].labels)
    preds.append(labels_pred.cpu().tolist())

    print('------------ Fold [{}/{}] ------------'.format(k, n_folds))
    print('Classification accuracy: {:.4f}'.format(acc_test))

targets = list(chain.from_iterable(targets))
preds = list(chain.from_iterable(preds))

print('-------------------------')
print('Classification accuracy: {:.4f}\u00B1{:.4f}'.format(numpy.mean(accs), numpy.std(accs)))
