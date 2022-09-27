import numpy
import torch
from util.data import load_dataset_fg
from mosgen.model_fgd import MOSGEN


# Ethanol
# smarts_target_fg = 'CCO'

# Butanone
# smarts_target_fg = 'O=C(C)CC'

# Naphthalene
# smarts_target_fg = 'c1ccc2ccccc2c1'

# Biphenyl
smarts_target_fg = 'c1ccc(cc1)-c1ccccc1'

n_folds = 3
n_epochs = 5000


dataset = load_dataset_fg('datasets/ir_dataset.json', smarts_target_fg)
kfolds = dataset.get_k_fold(k=n_folds)
accs = list()
f1_scores = list()
targets = list()
preds = list()


for k in range(0, n_folds):
    dataset_train = kfolds[k][0]
    dataset_test = kfolds[k][1]

    model = MOSGEN(dim_node_feats=dataset.dim_node_feats,
                   dim_edge_feats=dataset.dim_edge_feats).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-6)
    criterion = torch.nn.BCELoss()

    for epoch in range(0, n_epochs):
        loss_train, acc_train = model.fit(dataset_train, optimizer, criterion)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, n_epochs, loss_train))

    _, acc_test, f1_test = model.test(dataset_test)
    accs.append(acc_test)
    f1_scores.append(f1_test)

    print('------------ Fold [{}/{}] ------------'.format(k, n_folds))
    print('Detection accuracy: {:.4f}\tF1-score: {:.4f}'.format(acc_test, f1_test))

print('-------------------------')
print('Detection accuracy: {:.4f}\u00B1{:.4f}'.format(numpy.mean(accs), numpy.std(accs)))
print('F1-score: {:.4f}\u00B1{:.4f}'.format(numpy.mean(f1_scores), numpy.std(f1_scores)))
