import json
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import torch.nn.functional as F
import tqdm
import gc
import traceback
import sys
import torch_geometric
from sklearn.metrics import roc_auc_score
import psutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_memory_usage():
    print(f"CPU Memory: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

def load_jsonld(file_path):
    print("Loading JSON-LD data...")
    with open(file_path, 'r') as file:
        data = json.load(file)
    print("JSON-LD data loaded.")
    return data['assertion'] if 'assertion' in data else data

def create_hetero_data_from_jsonld(jsonld_data, link_type):
    print("Creating HeteroData...")
    data = HeteroData()
    node_types = set()
    edge_types = set()
    node_features = {}
    node_id_map = {}
    edge_index_dict = {}

    def clean_id(id_string):
        return id_string.rstrip('.0') if id_string.endswith('.0') else id_string

    # First pass: identify node types and create node mappings
    for item in jsonld_data:
        if '@type' in item:
            node_type = item['@type'][0].split('/')[-1]
            node_types.add(node_type)
            if node_type not in node_id_map:
                node_id_map[node_type] = {}
            if '@id' in item:
                node_id = clean_id(item['@id'])
                if node_id not in node_id_map[node_type]:
                    node_id_map[node_type][node_id] = len(node_id_map[node_type])

    # Second pass: extract features and create edges
    for item in jsonld_data:
        if '@type' in item:
            node_type = item['@type'][0].split('/')[-1]
            node_id = clean_id(item['@id'])
            node_idx = node_id_map[node_type][node_id]

            features = []
            for key, value in item.items():
                if key.startswith('http://schema.org/'):
                    if isinstance(value, list):
                        for v in value:
                            if isinstance(v, dict):
                                if '@value' in v:
                                    features.append(v['@value'])
                                elif '@id' in v:
                                    # This is a link to another entity
                                    target_id = clean_id(v['@id'])
                                    target_type = next((t for t in node_types if target_id in node_id_map[t]), None)
                                    if target_type:
                                        edge_type = (node_type, key.split('/')[-1], target_type)
                                        if edge_type not in edge_index_dict:
                                            edge_index_dict[edge_type] = [[], []]
                                        edge_index_dict[edge_type][0].append(node_idx)
                                        edge_index_dict[edge_type][1].append(node_id_map[target_type][target_id])
                                        edge_types.add(edge_type)

            if node_type not in node_features:
                node_features[node_type] = []
            node_features[node_type].append(features)

    # Create node feature tensors
    for node_type in node_types:
        if node_features.get(node_type):
            float_features = []
            for features in node_features[node_type]:
                float_feat = []
                for f in features:
                    try:
                        float_feat.append(float(f))
                    except ValueError:
                        float_feat.append(0.0)
                float_features.append(float_feat)
            
            max_features = max(len(f) for f in float_features)
            padded_features = [f + [0.0] * (max_features - len(f)) for f in float_features]
            data[node_type].x = torch.tensor(padded_features, dtype=torch.float)
        else:
            num_nodes = len(node_id_map[node_type])
            data[node_type].x = torch.ones((num_nodes, 1), dtype=torch.float)
        
        data[node_type].node_id = torch.arange(len(node_id_map[node_type]))

    # Create edge index tensors
    for edge_type, indices in edge_index_dict.items():
        data[edge_type].edge_index = torch.tensor(indices, dtype=torch.long)

    print("Node types:", node_types)
    print("Edge types:", edge_types)
    for edge_type, edge_index in data.edge_index_dict.items():
        print(f"Edge type {edge_type}: {edge_index.size()}")

    max_nodes = 10000
    for node_type in data.node_types:
        if data[node_type].num_nodes > max_nodes:
            perm = torch.randperm(data[node_type].num_nodes)[:max_nodes]
            data[node_type].x = data[node_type].x[perm]
            data[node_type].node_id = data[node_type].node_id[perm]
            for edge_type in data.edge_types:
                if edge_type[0] == node_type:
                    mask = data[edge_type].edge_index[0] < max_nodes
                    data[edge_type].edge_index = data[edge_type].edge_index[:, mask]
                if edge_type[2] == node_type:
                    mask = data[edge_type].edge_index[1] < max_nodes
                    data[edge_type].edge_index = data[edge_type].edge_index[:, mask]

    print(f"Number of nodes for each type: {[f'{node_type}: {len(node_id_map[node_type])}' for node_type in node_types]}")
    print(f"Number of edges for each type: {[f'{edge_type}: {len(edge_index_dict[edge_type][0])}' for edge_type in edge_types]}")

    if link_type in edge_index_dict:
        print(f"Found target edge type {link_type} in edge_index_dict")
        print(f"Number of edges: {len(edge_index_dict[link_type][0])}")
    else:
        print(f"Target edge type {link_type} not found in edge_index_dict")
        print("Available edge types:", data.edge_types)

    for edge_type in data.edge_types:
        if hasattr(data[edge_type], 'edge_index'):
            print(f"Edge type {edge_type} is properly configured with edge_index.")
        else:
            print(f"Edge type {edge_type} lacks edge_index.")

        if not hasattr(data[edge_type], 'edge_label'):
            print(f"Edge type {edge_type} lacks edge_label, adding default.")
            data[edge_type].edge_label = torch.zeros(data[edge_type].edge_index.size(1), dtype=torch.float)

    print("HeteroData created.")
    return data

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(2):  # Two layers
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels, normalize=True)
                for edge_type in metadata[1]
            }, aggr='mean')
            self.convs.append(conv)
        
        self.linear = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.linear[node_type] = Linear(-1, hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(F.dropout(x, p=0.5, training=self.training)) 
                      for key, x in x_dict.items()}
        return x_dict

class Classifier(torch.nn.Module):
    def __init__(self, link_type):
        super().__init__()
        self.link_type = link_type

    def forward(self, x_dict, edge_label_index):
        x_src = x_dict[self.link_type[0]][edge_label_index[0]]
        x_dst = x_dict[self.link_type[2]][edge_label_index[1]]
        return (x_src * x_dst).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data, link_type):
        super().__init__()
        self.embeddings = torch.nn.ModuleDict()
        for node_type in data.node_types:
            num_nodes = data[node_type].num_nodes
            self.embeddings[node_type] = torch.nn.Embedding(num_nodes, hidden_channels)
        self.gnn = GNN(hidden_channels, data.metadata())
        self.classifier = Classifier(link_type)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = {node_type: self.embeddings[node_type].weight for node_type in x_dict.keys()}
        x_dict = self.gnn(x_dict, edge_index_dict)
        return self.classifier(x_dict, edge_label_index)

def train_and_evaluate(jsonld_file, link_type):
    # Load and preprocess data
    print("Starting data load and preprocessing...")
    try:
        jsonld_data = load_jsonld(jsonld_file)
        data = create_hetero_data_from_jsonld(jsonld_data, link_type)
    except Exception as e:
        print(f"Error during data loading and preprocessing: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

    # Normalize features
    print("Normalizing features...")
    for node_type in data.node_types:
        if data[node_type].x is not None:
            mean = data[node_type].x.mean(dim=0, keepdim=True)
            std = data[node_type].x.std(dim=0, keepdim=True)
            data[node_type].x = (data[node_type].x - mean) / (std + 1e-5)

    # Add reverse edges to ensure all node types are updated
    print("Adding reverse edges...")
    for edge_type in list(data.edge_types):
        rev_edge_type = (edge_type[2], f'rev_{edge_type[1]}', edge_type[0])
        data[rev_edge_type].edge_index = data[edge_type].edge_index.flip([0])

    # Only process edge types with at least one edge
    valid_edge_types = [et for et in data.edge_types if data[et].num_edges > 0]

    if not valid_edge_types:
        print("No valid edge types found. Cannot proceed with link prediction.")
        sys.exit(1)

    # Split the data
    print("Splitting data into train, validation, and test sets...")
    num_edges = data[link_type].edge_index.size(1)
    num_val = max(1, int(0.1 * num_edges))
    num_test = max(1, int(0.1 * num_edges))

    perm = torch.randperm(num_edges)
    val_edges = perm[:num_val]
    test_edges = perm[num_val:num_val+num_test]
    train_edges = perm[num_val+num_test:]

    # Create train, validation, and test data
    train_data = data.clone()
    val_data = data.clone()
    test_data = data.clone()

    train_data[link_type].edge_label_index = data[link_type].edge_index[:, train_edges]
    train_data[link_type].edge_label = torch.ones(train_edges.size(0))

    val_data[link_type].edge_label_index = data[link_type].edge_index[:, val_edges]
    val_data[link_type].edge_label = torch.ones(val_edges.size(0))

    test_data[link_type].edge_label_index = data[link_type].edge_index[:, test_edges]
    test_data[link_type].edge_label = torch.ones(test_edges.size(0))

    # Create negative samples for validation and test sets
    num_neg_samples = num_val
    neg_edge_index = torch.randint(0, data[link_type[0]].num_nodes, (2, num_neg_samples), dtype=torch.long)
    val_data[link_type].edge_label_index = torch.cat([val_data[link_type].edge_label_index, neg_edge_index], dim=1)
    val_data[link_type].edge_label = torch.cat([val_data[link_type].edge_label, torch.zeros(num_neg_samples)])

    num_neg_samples = num_test
    neg_edge_index = torch.randint(0, data[link_type[0]].num_nodes, (2, num_neg_samples), dtype=torch.long)
    test_data[link_type].edge_label_index = torch.cat([test_data[link_type].edge_label_index, neg_edge_index], dim=1)
    test_data[link_type].edge_label = torch.cat([test_data[link_type].edge_label, torch.zeros(num_neg_samples)])

    # LinkNeighborLoader setup
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[10, 5],
        neg_sampling_ratio=1.0,
        edge_label_index=(link_type, train_data[link_type].edge_label_index),
        edge_label=train_data[link_type].edge_label,
        batch_size=128,
        shuffle=True,
    )

    # Initialize model and optimizer
    model = Model(hidden_channels=64, data=data, link_type=link_type)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Training loop
    print("Start training loop...")
    for epoch in range(1, 61):
        model.train()
        total_loss = total_examples = 0
        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            try:
                batch = batch.to(device)
                pred = model(batch.x_dict, batch.edge_index_dict, batch[link_type].edge_label_index)
                ground_truth = batch[link_type].edge_label
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
            except Exception as e:
                print(f"Error in batch: {str(e)}")
                traceback.print_exc()
                continue
        
        if total_examples > 0:
            print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
        else:
            print(f"Epoch: {epoch:03d}, No valid examples")
        
        gc.collect
        print_memory_usage()
        torch.cuda.empty_cache()

    # Evaluation
    model.eval()

    val_loader = LinkNeighborLoader(
        val_data, 
        num_neighbors=[10, 5], 
        batch_size=128,
        edge_label_index=(link_type, val_data[link_type].edge_label_index),
        edge_label=val_data[link_type].edge_label,
        shuffle=False
    )

    val_predictions = []
    val_ground_truth = []

    print("Starting validation...")
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc="Validating"):
            try:
                batch = batch.to(device)
                pred = model(batch.x_dict, batch.edge_index_dict, batch[link_type].edge_label_index)
                ground_truth = batch[link_type].edge_label
                val_predictions.append(pred.cpu())
                val_ground_truth.append(ground_truth.cpu())
            except Exception as e:
                print(f"Error in validation batch: {str(e)}")
                traceback.print_exc()
                continue

            gc.collect()
            print_memory_usage()

        print("Validation loop completed")

    # Calculate validation metrics
    if val_predictions and val_ground_truth:
        val_predictions = torch.cat(val_predictions)
        val_ground_truth = torch.cat(val_ground_truth)
        
        print(f"Total validation samples: {len(val_ground_truth)}")
        print(f"Positive samples: {val_ground_truth.sum().item()}")
        print(f"Negative samples: {len(val_ground_truth) - val_ground_truth.sum().item()}")
        
        if len(torch.unique(val_ground_truth)) > 1:
            val_auc = roc_auc_score(val_ground_truth.numpy(), val_predictions.numpy())
            print(f"Validation AUC: {val_auc:.4f}")
        else:
            print("Warning: Only one class present in validation set. Cannot calculate AUC.")
    else:
        print("No valid predictions or ground truth for validation")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <jsonld_file> <link_type>")
        print("Example: python script.py data.jsonld '(InvestmentOrGrant,investee,Organization)'")
        sys.exit(1)

    jsonld_file = sys.argv[1]
    link_type = eval(sys.argv[2])  # Convert string to tuple
    train_and_evaluate(jsonld_file, link_type)
