import dgl
import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm
from dgllife.model.model_zoo import GCNPredictor
from graphein.construct_graphs import ProteinGraph
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from dgl.data.utils import save_graphs, load_graphs, load_labels

from matplotlib import pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Load data sets
df = pd.read_csv('labels_reduced_dropped.csv')
df.head()
#df = df.iloc[:15]
print(df)

# Create labels
labels = pd.get_dummies(df.EC).values.tolist()
print("printing labels")
print(labels)
labels = [torch.Tensor(i) for i in labels]

# Split datasets
x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.15)

print(y_test)
"""
class GraphLoader():

    def __init__(self):
        # Initialise Graph Constructor
        self.pg = ProteinGraph(granularity='CA', insertions=False, keep_hets=True,
                  node_featuriser='meiler',
                  get_contacts_path='~/projects/repos/getcontacts',
                  pdb_dir='files/pdbs/',
                  contacts_dir='files/contacts/',
                  exclude_waters=True, covalent_bonds=True, include_ss=True)

    def get_graph(self, pdb_id, label):

        file_path = "files/graphs/" + pdb_id + ".bin"
        print(file_path)
        try:
            g = load_graphs(file_path)
        except:
            print("Didn't find precomputed graph, generating now")
            g = self.pg.dgl_graph_from_pdb_code(pdb_code=pdb_id)
            print(g)
            print(label)
            graph_label = {"glabel": torch.tensor(0)}
            save_graphs(file_path, g)
            print("Graph generated and saved")
        return g

gl = GraphLoader()

# Build Graphs


train_graphs = [gl.get_graph(x_train["pdb_id"].iloc[i], y_train[i]) for i in tqdm(range(len(x_train)))]

test_graphs = [gl.get_graph(x_test["pdb_id"].iloc[i], y_test[i]) for i in tqdm(range(len(x_test)))]


"""
pg = ProteinGraph(granularity='CA', insertions=False, keep_hets=True,
          node_featuriser='meiler',
          get_contacts_path='~/projects/repos/getcontacts',
          pdb_dir='files/pdbs/',
          contacts_dir='files/contacts/',
          exclude_waters=True,
          covalent_bonds=True,
          include_ss=True,
          include_ligand=False,
          remove_string_labels=True)
# Build Graphs
"""
train_graphs = [pg.dgl_graph_from_pdb_code(pdb_code=x_train["pdb_id"].iloc[i]) for i in tqdm(range(len(x_train)))]

test_graphs = [pg.dgl_graph_from_pdb_code(pdb_code=x_test["pdb_id"].iloc[i]) for i in tqdm(range(len(x_test)))]

#save_graphs("train_graphs.bin", train_graphs)
#save_graphs("test_graphs.bin", test_graphs)

pickle.dump(train_graphs, open( "train_graphs.p", "wb"))
pickle.dump(test_graphs, open( "test_graphs.p", "wb" ) )
"""
print("Loading graphs")

with open("train_graphs.p") as pickle_file:
    train_graphs = pickle.load(pickle_file)

with open("test_graphs.p") as pickle_file:
    test_graphs = pickle.load(pickle_file)

print("Loaded graphs")

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs, node_attrs='h')
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph, torch.stack(labels)

train_data = list(zip(train_graphs, y_train))
test_data = list(zip(test_graphs, y_test))

#Create dataloaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                         collate_fn=collate)

test_loader = DataLoader(test_data, batch_size=32, shuffle=True,
                         collate_fn=collate)



n_feats = train_graphs[1].ndata['h'].shape[1]

print(train_graphs[1].edata["rel_type"].shape)
print(train_graphs[1].edata["norm"].shape)
print("-------------")
exit()

# Instantiate model
gcn_net = GCNPredictor(in_feats=n_feats,
                       hidden_feats=[32, 32],
                       batchnorm=[True, True],
                       dropout=[0, 0],
                       classifier_hidden_feats=32,
                       n_tasks=6
                       )
gcn_net.to(device)
loss_fn = CrossEntropyLoss()
optimizer = torch.optim.Adam(gcn_net.parameters(), lr=0.005)

epochs = 20

# Training loop
gcn_net.train()
epoch_losses = []

epoch_f1_scores = []
epoch_precision_scores = []
epoch_recall_scores = []

for epoch in range(epochs):
    epoch_loss = 0

    preds = []
    labs = []
    # Train on batch
    for i, (bg, labels) in enumerate(train_loader):
        labels = labels.to(device)
        graph_feats = bg.ndata.pop('h').to(device)
        graph_feats, labels = graph_feats.to(device), labels.to(device)
        y_pred = gcn_net(bg, graph_feats)

        preds.append(y_pred.detach().numpy())
        labs.append(labels.detach().numpy())

        labels = np.argmax(labels, axis=1)

        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

    epoch_loss /= (i + 1)

    preds = np.vstack(preds)
    labs = np.vstack(labs)

    # There's some sort of issue going on here with the scoring. All three values are the same
    f1 = f1_score(np.argmax(labs, axis=1), np.argmax(preds, axis=1), average='micro')
    precision = precision_score(np.argmax(labs, axis=1), np.argmax(preds, axis=1), average='micro')
    recall = recall_score(np.argmax(labs, axis=1), np.argmax(preds, axis=1), average='micro')

    if epoch % 5 == 0:
        print(f"epoch: {epoch}, LOSS: {epoch_loss:.3f}, F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

    epoch_losses.append(epoch_loss)
    epoch_f1_scores.append(f1)
    epoch_precision_scores.append(precision)
    epoch_recall_scores.append(recall)

# Evaluate
gcn_net.eval()
test_loss = 0

preds = []
labs = []
for i, (bg, labels) in enumerate(test_loader):
    labels = labels.to(device)
    graph_feats = bg.ndata.pop('h').to(device)
    graph_feats, labels = graph_feats.to(device), labels.to(device)
    y_pred = gcn_net(bg, graph_feats)

    preds.append(y_pred.detach().numpy())
    labs.append(labels.detach().numpy())

labs = np.vstack(labs)
preds = np.vstack(preds)

f1 = f1_score(np.argmax(labs, axis=1), np.argmax(preds, axis=1), average='micro')
precision = precision_score(np.argmax(labs, axis=1), np.argmax(preds, axis=1), average='micro')
recall = recall_score(np.argmax(labs, axis=1), np.argmax(preds, axis=1), average='micro')

print(f"TEST F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
