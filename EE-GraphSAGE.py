# import dgl.nn as dglnn
import torch
from dgl import from_networkx
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl.function as fn
import networkx as nx
import pandas as pd
import socket
import struct
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler




data = pd.read_csv('data-col35.csv')
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])
le.fit(data.Label.values)
print("统计每个标签的数量：")
label_counts = data['Label'].value_counts()
print("Original label counts:")
print(label_counts)

# import sys
# sys.exit()

# data['src_ip'] = data.src_ip.apply(
#     lambda x: socket.inet_ntoa(struct.pack('>I', random.randint(0xac100001, 0xac1f0001))))

data['IP_src'] = data.IP_src.apply(
    lambda x: socket.inet_ntoa(struct.pack('>I', random.randint(0xac100001, 0xac1f0001))))

data['IP_src'] = data.IP_src.apply(str)
data['IP_dst'] = data.IP_dst.apply(str)

# 删除IP地址列中为0的行
# index_to_drop = data[data['IP_dst'] == '0'].index
# data.drop(index_to_drop, inplace=True)

labels = data["Label"]
features = data.drop(columns=["Label"])
features = features.drop(['IP_src', 'IP_dst'], axis=1)

# 将除IP地址以外的剩下的特征转换为 NumPy 数组
features_array = features.values.astype(np.float32)
labels_array = labels.values.astype(np.int64)  # 如果标签是整数型，可以使用 np.long 类型

# 数据标准化归一化
scaler = StandardScaler()
standardized_data = scaler.fit_transform(features_array)
min_max_scaler = MinMaxScaler()
normalized_data = min_max_scaler.fit_transform(standardized_data)

# 需要将它们转换为 Tensor 类型
features_tensor = torch.tensor(normalized_data, dtype=torch.float32)
labels_tensor = torch.tensor(labels_array, dtype=torch.long)  # 如果标签是整数型，可以使用 torch.long 类型


features_tensor = pd.concat([data[['IP_src', 'IP_dst']].reset_index(drop=True), pd.DataFrame(features_tensor.numpy())], axis=1)

#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    features_tensor, labels_tensor, test_size=0.3, random_state=42, stratify=labels_tensor)
train_ds = len(X_train)
val_ds = len(X_test)


cols_to_norm = list(set(list(X_train.iloc[:, 2:].columns)) - set(list(['label'])))
X_train[cols_to_norm] = scaler.fit_transform(X_train[cols_to_norm])


X_train['label'] = y_train
X_train['h'] = X_train[cols_to_norm].values.tolist()

G = nx.from_pandas_edgelist(X_train, "IP_src", "IP_dst", ['h', 'label'], create_using=nx.MultiGraph())
G = G.to_directed()
G = from_networkx(G, edge_attrs=['h', 'label'])

# Eq1
G.ndata['h'] = th.ones(G.num_nodes(), G.edata['h'].shape[1])
G.ndata['h'] = th.reshape(G.ndata['h'], (G.ndata['h'].shape[0], 1, G.ndata['h'].shape[1]))
G.edata['h'] = th.reshape(G.edata['h'], (G.edata['h'].shape[0], 1, G.edata['h'].shape[1]))

G.edata['train_mask'] = th.ones(len(G.edata['h']), dtype=th.bool)

class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super().__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        # self.pred = MLPPredictor(ndim_out, 10)
        self.pred = FullyConnectedPredictor(ndim_out, 8)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)

class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation, residual=True):
        super(SAGELayer, self).__init__()
        self.residual = residual
        ### force to outut fix dimensions
        self.W_msg111 = nn.Linear(32, 32)
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        ### apply weight
        self.W_apply = nn.Linear(128, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        return {'m': self.W_msg(th.cat([edges.src['h'], edges.data['h']], 2))}
    def message_func111(self, edges):
        return {'m': self.W_msg111(th.add(edges.data['h'], edges.data['h']))}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats

            #根据边更新节点
            g.update_all(self.message_func111, fn.mean('m', 'h_neigh'))  #m是消息聚合的结果  h_neigh是执行平均聚合之后的结果
            g.ndata['h'] = F.relu(g.ndata['h_neigh'])

            #根据邻居节点和边更新节点
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            h_self = g.ndata['h']
            h_neigh = g.ndata['h_neigh']
            aaa = th.add(h_neigh, h_neigh)
            g.ndata['h'] = F.relu(self.W_apply(aaa))

            return g.ndata['h']

class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGELayer(ndim_in, edim, 128, activation))
        # self.layers.append(SAGELayer(128, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)



class FullyConnectedPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.fc111 = nn.Linear(32, 128)
        self.fc = nn.Linear(256, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h'] * 10
        h_v = edges.dst['h'] * 10
        concatenated = th.add(h_u, h_v)
        edges.data['h'] = torch.squeeze(edges.data['h'], dim=1)
        edges.data['h'] = self.fc111(edges.data['h'])
        concatenated = th.cat([concatenated, edges.data['h']], 1)
        score = self.fc(concatenated)
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=np.unique(G.edata['label'].cpu().numpy()),
                                                  y=G.edata['label'].cpu().numpy())

class_weights = th.FloatTensor(class_weights)
criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss(weight=class_weights)


def compute_accuracy(pred, labels):
    return (pred.argmax(1) == labels).float().mean().item()


import os

node_features = G.ndata['h']
edge_features = G.edata['h']

edge_label = G.edata['label']
train_mask = G.edata['train_mask']

model = Model(G.ndata['h'].shape[2], 128, G.ndata['h'].shape[2], F.relu, 0.2)
opt = th.optim.Adam(model.parameters())


losses = []
accuracies = []

for epoch in range(1, 100):
    pred = model(G, node_features, edge_features)
    loss = criterion(pred[train_mask], edge_label[train_mask])
    opt.zero_grad()
    loss.backward()
    opt.step()

    # 记录损失和准确率
    losses.append(loss.item())
    acc = compute_accuracy(pred[train_mask], edge_label[train_mask])
    accuracies.append(acc)

    if epoch % 1 == 0:
        # if epoch % 100 == 0:
        #     th.save(model.state_dict(), os.path.join('./model', 'epoch-{}.pt'.format(epoch)))
        print('Epoch:', epoch, ' Training acc:', compute_accuracy(pred[train_mask], edge_label[train_mask]))



X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])

X_test['label'] = y_test
X_test['h'] = X_test[cols_to_norm].values.tolist()

G_test = nx.from_pandas_edgelist(X_test, "IP_src", "IP_dst", ['h', 'label'], create_using=nx.MultiGraph())
G_test = G_test.to_directed()
G_test = from_networkx(G_test, edge_attrs=['h', 'label'])
actual = G_test.edata.pop('label')
G_test.ndata['feature'] = th.ones(G_test.num_nodes(), G.ndata['h'].shape[2])

G_test.ndata['feature'] = th.reshape(G_test.ndata['feature'],
                                     (G_test.ndata['feature'].shape[0], 1, G_test.ndata['feature'].shape[1]))
G_test.edata['h'] = th.reshape(G_test.edata['h'], (G_test.edata['h'].shape[0], 1, G_test.edata['h'].shape[1]))

import timeit

start_time = timeit.default_timer()
node_features_test = G_test.ndata['feature']
edge_features_test = G_test.edata['h']
test_pred = model(G_test, node_features_test, edge_features_test)
elapsed = timeit.default_timer() - start_time

print(str(elapsed) + ' seconds')

test_pred = test_pred.argmax(1)
test_pred = th.Tensor.cpu(test_pred).detach().numpy()


print("输出混淆矩阵")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(actual, test_pred)
print(cm)


print("计算评价指标")
import numpy as np
# from tabulate import tabulate

conf_matrix = cm

# 计算评估指标
accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
recall_per_class = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
recall = np.mean(recall_per_class)
precision_per_class = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
class_counts = np.sum(conf_matrix, axis=1)
weighted_f1 = np.sum(f1_per_class * class_counts) / np.sum(class_counts)

table_data = []
for i in range(len(recall_per_class)):
    table_data.append([f"类别 {i}",
                       np.round(np.diag(conf_matrix)[i] / np.sum(conf_matrix[i]), 3),
                       np.round(recall_per_class[i], 3),
                       np.round(precision_per_class[i], 3),
                       np.round(f1_per_class[i], 3)])

table_data.append(["平均", np.round(accuracy, 3), np.round(recall, 3), np.round(np.mean(precision_per_class), 3),
                   np.round(weighted_f1, 3)])

headers = ["类别", "准确率", "召回率", "精确率", "加权平均F1值"]
# print(tabulate(table_data, headers=headers, floatfmt=".3f", tablefmt="pretty", numalign='center'))

print('over')

# from sklearn.metrics import plot_confusion_matrix

import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


from sklearn.metrics import confusion_matrix

print("target_names",np.unique(actual))
plot_confusion_matrix(cm=confusion_matrix(actual, test_pred),
                      normalize=False,
                      target_names=np.unique(actual),
                      title="Confusion Matrix")

from sklearn.metrics import classification_report
print(classification_report(actual, test_pred, digits=4))
report = classification_report(actual, test_pred, digits=4, output_dict=True)
df = pd.DataFrame(report).transpose()




