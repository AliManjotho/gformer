import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool

class GraphTransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads):
        super(GraphTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        # Create Transformer layers
        for _ in range(num_layers):
            self.layers.append(TransformerConv(input_dim, hidden_dim, heads=num_heads, concat=False))
            input_dim = hidden_dim

        # Final linear layer for classification
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Pass through each Transformer layer
        for layer in self.layers:
            x = layer(x, edge_index)
            x = torch.relu(x)

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Classification
        out = self.classifier(x)
        return out




