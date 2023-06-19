import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, SAGEConv
from torch.nn import Sequential, Linear, ReLU





class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="max")  # "Max" aggregation.
        self.mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        return self.propagate(edge_index, x=x)  # shape [num_nodes, out_channels]

    def message(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        # x_j: Source node features of shape [num_edges, in_channels]
        # x_i: Target node features of shape [num_edges, in_channels]
        edge_features = torch.cat([x_i, x_j - x_i], dim=-1)
        return self.mlp(edge_features)  # shape [num_edges, out_channels]


class GNN_ADD(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, int(hidden_channels/4))
        self.conv4 = GCNConv(int(hidden_channels/4), int(hidden_channels/4))
        self.conv5 = GCNConv(int(hidden_channels/4), out_channels)
        self.lin = Linear(out_channels,1)

    def forward(self, x, edge_index, batch_size):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges] 
        x = x.view(-1,2) #batch_size*num_nodes, 2   
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index) 
        x = F.relu(x)
        x = self.conv4(x, edge_index) 
        x = F.relu(x)
        x = self.conv5(x, edge_index) 
        x = F.relu(x)

        #x = F.dropout(x, training=self.training)

        x= self.lin(x) #batch_size*num_nodes, 1
        x= x.view(batch_size, -1) 

        return x

class GNN_LIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = Linear(in_channels, hidden_channels)
        self.conv2 = Linear(hidden_channels, hidden_channels)
        self.conv3 = Linear(hidden_channels, int(hidden_channels/4))
        self.conv4 = Linear(int(hidden_channels/4), int(hidden_channels/4))
        self.conv5 = Linear(int(hidden_channels/4), out_channels)
        self.lin = Linear(out_channels,1)

    def forward(self, x, edge_index, batch_size):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges] 
        x = x.view(-1,2) #batch_size*num_nodes, 2   
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x) 
        x = F.relu(x)
        x = self.conv4(x) 
        x = F.relu(x)
        x = self.conv5(x) 
        x = F.relu(x)

        x = F.dropout(x, training=self.training)

        x= self.lin(x) #batch_size*num_nodes, 1
        x= x.view(batch_size, -1) 

        return x

class GNN_MEAN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, int(hidden_channels/4))
        self.conv4 = SAGEConv(int(hidden_channels/4), int(hidden_channels/4))
        self.conv5 = SAGEConv(int(hidden_channels/4), out_channels)
        self.lin = Linear(out_channels,1)

    def forward(self, x, edge_index, batch_size):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges] 
        x = x.view(-1,2) #batch_size*num_nodes, 2   
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index) 
        x = F.relu(x)
        x = self.conv4(x, edge_index) 
        x = F.relu(x)
        x = self.conv5(x, edge_index) 
        x = F.relu(x)

        x = F.dropout(x, training=self.training)

        x= self.lin(x) #batch_size*num_nodes, 1
        x= x.view(batch_size, -1) 

        return x


class GNN_MAX(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = EdgeConv(in_channels, hidden_channels)
        self.conv2 = EdgeConv(hidden_channels, hidden_channels)
        self.conv3 = EdgeConv(hidden_channels, out_channels)
        self.lin = Linear(out_channels,1)

    def forward(self, x, edge_index, batch_size):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges] 
        x = x.view(-1,2) #batch_size*num_nodes, 2   
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index) 

        x = F.dropout(x, training=self.training)

        x= self.lin(x) #batch_size*num_nodes, 1
        x= x.view(batch_size, -1) 

        return x
