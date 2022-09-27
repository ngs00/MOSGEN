import numpy
import torch
from util.data import load_dataset
from mosgen.model import MOSGEN


n_folds = 3
n_epochs = 5000
n_suggestions = 3


dataset = load_dataset('datasets/ir_dataset.json')
kfolds = dataset.get_k_fold(k=n_folds, random_seed=0)
accs = list()


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

    query_values = [d.query_value.item() for d in dataset_test.dataset]
    comp_ids = dataset_test.search(labels_pred.cpu().tolist(), query_values, k=n_suggestions)
    acc_classes = numpy.zeros(dataset.n_classes)

    for i in range(0, len(dataset_test.dataset)):
        if dataset_test.comp_names[i] in comp_ids[i]:
            acc_classes[dataset_test.labels[i]] += 1

    for i in range(0, dataset.n_classes):
        acc_classes[i] = (100 * acc_classes[i]) / len(dataset_test.data_dist[i])

    print('------------ Fold [{}/{}] ------------'.format(k, n_folds))
    for i in range(0, dataset.n_classes):
        print('Identification accuracy of the compounds in {}: {:.4f}'.format(dataset.class_names[i], acc_classes[i]))
