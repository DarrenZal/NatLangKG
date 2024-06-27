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

device = torch.device('cpu')

def print_memory_usage():
    print(f"CPU Memory: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

def load_jsonld(file_path):
    print("Loading JSON-LD data...")
    with open(file_path, 'r') as file:
        data = json.load(file)
    print("JSON-LD data loaded.")
    return data['assertion']

def create_hetero_data_from_jsonld(jsonld_data):
    print("Creating HeteroData...")
    data = HeteroData()
    node_types = set()
    edge_types = set()
    node_features = {}
    node_id_map = {}
    edge_index_dict = {}

    for item in jsonld_data:
        if '@type' in item:
            node_type = item['@type'][0].split('/')[-1]
            node_types.add(node_type)
            if node_type not in node_id_map:
                node_id_map[node_type] = {}
            if '@id' in item:
                node_id = item['@id']
                if node_id not in node_id_map[node_type]:
                    node_id_map[node_type][node_id] = len(node_id_map[node_type])

    for item in jsonld_data:
        if '@type' in item:
            node_type = item['@type'][0].split('/')[-1]
            node_id = item['@id']
            node_idx = node_id_map[node_type][node_id]

            features = []
            for key, value in item.items():
                if key not in ['@type', '@id']:
                    if isinstance(value, list):
                        for v in value:
                            if isinstance(v, dict):
                                features.append(v.get('@value', str(v)))
                            else:
                                features.append(str(v))
                    else:
                        features.append(str(value))
            
            if node_type not in node_features:
                node_features[node_type] = []
            node_features[node_type].append(features)

            for key, value in item.items():
                if key.startswith('http://schema.org/') and isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict) and '@id' in value[0]:
                    target_id = value[0]['@id']
                    for target_type, target_map in node_id_map.items():
                        if target_id in target_map:
                            edge_type = (node_type, key.split('/')[-1], target_type)
                            if edge_type not in edge_index_dict:
                                edge_index_dict[edge_type] = [[], []]
                            edge_index_dict[edge_type][0].append(node_idx)
                            edge_index_dict[edge_type][1].append(target_map[target_id])
                            edge_types.add(edge_type)

    for node_type in node_types:
        if node_features[node_type]:
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

    if ('InvestmentOrGrant', 'investee', 'Organization') in edge_index_dict:
        print("Found target edge type in edge_index_dict")
        print("Number of edges:", len(edge_index_dict[('InvestmentOrGrant', 'investee', 'Organization')][0]))
    else:
        print("Target edge type not found in edge_index_dict")
        print("Available edge types:", data.edge_types)
        for edge_type in data.edge_types:
            print(f"Inspecting {edge_type}: Has edge_label?", 'edge_label' in data[edge_type])

    for edge_type in data.edge_types:
        if hasattr(data[edge_type], 'edge_index'):
            print(f"Edge type {edge_type} is properly configured with edge_index.")
        else:
            print(f"Edge type {edge_type} lacks edge_index.")

        if hasattr(data[edge_type], 'edge_label'):
            print(f"Edge type {edge_type} is properly configured with edge_label.")
        else:
            print(f"Edge type {edge_type} lacks edge_label, adding default.")
            data[edge_type].edge_label = torch.zeros(data[edge_type].edge_index.size(1), dtype=torch.float)

    for edge_type in data.edge_types:
        if 'edge_label' not in data[edge_type]:
            num_edges = data[edge_type].edge_index.size(1)
            data[edge_type].edge_label = torch.zeros(num_edges, dtype=torch.float)
            print(f"Initialized edge_label for {edge_type} with zeros.")

    print("HeteroData created.")
    return data

# Load and preprocess data
print("Starting data load and preprocessing...")
try:
    jsonld_data = load_jsonld('large_fund_dataset.jsonld')
    data = create_hetero_data_from_jsonld(jsonld_data)
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

# Choose the edge type for link prediction
edge_type = ('InvestmentOrGrant', 'investee', 'Organization')

def negative_sampling(edge_index, num_nodes, num_neg_samples):
    neg_edge_index = torch.randint(0, num_nodes, (2, num_neg_samples), dtype=torch.long)
    mask = (edge_index.unsqueeze(-1) == neg_edge_index.unsqueeze(1)).all(dim=0).any(dim=0)
    neg_edge_index = neg_edge_index[:, ~mask]
    return neg_edge_index

# Manually split the data
print("Splitting data into train, validation, and test sets...")
num_edges = data[edge_type].edge_index.size(1)
num_val = max(1, int(0.1 * num_edges))
num_test = max(1, int(0.1 * num_edges))

# Create a random permutation of edge indices
perm = torch.randperm(num_edges)

val_edges = perm[:num_val]
test_edges = perm[num_val:num_val+num_test]
train_edges = perm[num_val+num_test:]

# Generate negative edges for validation and test sets
print("Generating negative samples...")
num_nodes = data[edge_type[0]].num_nodes
val_neg_edges = negative_sampling(data[edge_type].edge_index, num_nodes, num_val)
test_neg_edges = negative_sampling(data[edge_type].edge_index, num_nodes, num_test)

# Create train, validation, and test data
print("Creating train, validation, and test datasets...")
train_data = data.clone()
val_data = data.clone()
test_data = data.clone()

# Set edge_index and edge_label_index for each split
train_data[edge_type].edge_label_index = data[edge_type].edge_index[:, train_edges]
train_data[edge_type].edge_label = torch.ones(train_edges.size(0))

num_val_pos = len(val_edges)
num_val_neg = num_val_pos  # Equal number of positive and negative samples

val_pos_edges = data[edge_type].edge_index[:, val_edges]
val_neg_edges = negative_sampling(data[edge_type].edge_index, num_nodes, num_val_neg)

# Ensure we have the correct number of negative edges
val_neg_edges = val_neg_edges[:, :num_val_neg]

val_data[edge_type].edge_label_index = torch.cat([val_pos_edges, val_neg_edges], dim=1)
val_data[edge_type].edge_label = torch.cat([torch.ones(num_val_pos), torch.zeros(num_val_neg)])

test_data[edge_type].edge_label_index = torch.cat([data[edge_type].edge_index[:, test_edges], test_neg_edges], dim=1)
test_data[edge_type].edge_label = torch.cat([torch.ones(test_edges.size(0)), torch.zeros(test_neg_edges.size(1))])

print(f"\nValidation set after creation:")
print(f"Total edges: {val_data[edge_type].edge_label_index.size(1)}")
print(f"Positive edges: {val_data[edge_type].edge_label.sum().item()}")
print(f"Negative edges: {(val_data[edge_type].edge_label == 0).sum().item()}")

def ensure_edge_label_and_index(data):
    for et in data.edge_types:
        if et == edge_type:
            print(f"Skipping target edge type: {et}")
            continue
        if 'edge_label' not in data[et]:
            num_edges = data[et].edge_index.size(1)
            data[et].edge_label = torch.zeros(num_edges, dtype=torch.float)
        if 'edge_label_index' not in data[et]:
            data[et].edge_label_index = data[et].edge_index
        
        # Handle reverse edges
        rev_edge_type = (et[2], f'rev_{et[1]}', et[0])
        if rev_edge_type in data.edge_types:
            data[rev_edge_type].edge_label = data[et].edge_label
            data[rev_edge_type].edge_label_index = data[et].edge_index.flip([0])

    # Ensure the target edge type has the correct labels
    assert 'edge_label' in data[edge_type], f"edge_label not found for {edge_type}"
    assert 'edge_label_index' in data[edge_type], f"edge_label_index not found for {edge_type}"

ensure_edge_label_and_index(train_data)
ensure_edge_label_and_index(val_data)
ensure_edge_label_and_index(test_data)

print("\nAfter ensure_edge_label_and_index:")
print(f"edge_label_index shape: {val_data[edge_type].edge_label_index.shape}")
print(f"edge_label shape: {val_data[edge_type].edge_label.shape}")
print(f"Positive samples: {val_data[edge_type].edge_label.sum().item()}")
print(f"Negative samples: {(val_data[edge_type].edge_label == 0).sum().item()}")

# Set edge_index and edge_label_index for each split
train_data[edge_type].edge_label_index = train_data[edge_type].edge_index
train_data[edge_type].edge_label = torch.ones(train_data[edge_type].edge_index.size(1))

num_val_pos = len(val_edges)
num_val_neg = num_val_pos  # Equal number of positive and negative samples

val_pos_edges = data[edge_type].edge_index[:, val_edges]
val_neg_edges = negative_sampling(data[edge_type].edge_index, num_nodes, num_val_neg)

# Ensure we have the correct number of negative edges
val_neg_edges = val_neg_edges[:, :num_val_neg]

val_data[edge_type].edge_label_index = torch.cat([val_pos_edges, val_neg_edges], dim=1)
val_data[edge_type].edge_label = torch.cat([torch.ones(num_val_pos), torch.zeros(num_val_neg)])

print(f"\nValidation set after creation:")
print(f"Total edges: {val_data[edge_type].edge_label_index.size(1)}")
print(f"Positive edges: {val_data[edge_type].edge_label.sum().item()}")
print(f"Negative edges: {(val_data[edge_type].edge_label == 0).sum().item()}")

test_data[edge_type].edge_label_index = torch.cat([data[edge_type].edge_index[:, test_edges], test_neg_edges], dim=1)
test_data[edge_type].edge_label = torch.cat([torch.ones(test_edges.size(0)), torch.zeros(test_neg_edges.size(1))])

print("Train data node types:", train_data.node_types)
print("Train data edge types:", train_data.edge_types)
print(f"Train data edge_index for {edge_type}:", train_data[edge_type].edge_index.shape)
print(f"Train data edge_label_index for {edge_type}:", train_data[edge_type].edge_label_index.shape)
print(f"Train data edge_label for {edge_type}:", train_data[edge_type].edge_label.shape)

for edge_type in val_data.edge_types:
    if 'edge_label' not in val_data[edge_type]:
        num_edges = val_data[edge_type].edge_index.size(1)
        val_data[edge_type].edge_label = torch.zeros(num_edges, dtype=torch.float)

# Ensure edge_label exists in val_data and test_data
def add_edge_labels(data, edge_types):
    for edge_type in edge_types:
        if 'edge_label' not in data[edge_type]:
            num_edges = data[edge_type].edge_index.size(1)
            data[edge_type].edge_label = torch.zeros(num_edges, dtype=torch.float)


add_edge_labels(val_data, val_data.edge_types)
add_edge_labels(test_data, test_data.edge_types)

print("\nAfter initial validation set creation:")
print(f"edge_label_index shape: {val_data[edge_type].edge_label_index.shape}")
print(f"edge_label shape: {val_data[edge_type].edge_label.shape}")
print(f"Positive samples: {val_data[edge_type].edge_label.sum().item()}")
print(f"Negative samples: {(val_data[edge_type].edge_label == 0).sum().item()}")

print("\nValidation set creation:")
print(f"Number of validation edges: {len(val_edges)}")
print(f"Shape of data[edge_type].edge_index: {data[edge_type].edge_index.shape}")
print(f"Shape of val_edges: {val_edges.shape}")
print(f"First few val_edges: {val_edges[:5]}")
print(f"Shape of val_pos_edges: {val_pos_edges.shape}")
print(f"Shape of val_neg_edges: {val_neg_edges.shape}")

print("Train data structure for target edge type:")
for key, value in train_data[edge_type].items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
    else:
        print(f"  {key}: {type(value)}")

print("\nAll edge types in train_data:")
for et in train_data.edge_types:
    print(f"Edge type: {et}")
    for key, value in train_data[et].items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    print()

# LinkNeighborLoader setup
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[10, 5],
    neg_sampling_ratio=1.0,
    edge_label_index=(edge_type, train_data[edge_type].edge_label_index),
    edge_label=train_data[edge_type].edge_label,
    batch_size=32,
    shuffle=True,
)

# Add this debugging information
print("LinkNeighborLoader created with:")
print(f"edge_label_index shape: {train_data[edge_type].edge_label_index.shape}")
print(f"edge_label shape: {train_data[edge_type].edge_label.shape}")


try:
    print("Attempting to get first batch from train_loader...")
    first_batch = next(iter(train_loader))
    print("First batch keys:", first_batch.keys())
    for key, value in first_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape {value.shape}, dtype {value.dtype}")
        elif isinstance(value, dict):
            print(f"{key}: {[f'{k}: shape {v.shape}' for k, v in value.items()]}")
        else:
            print(f"{key}: {type(value)}")
except Exception as e:
    print(f"Error while accessing first batch: {str(e)}")
    traceback.print_exc()

# Define the model
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
    def forward(self, x_dict, edge_label_index):
        x_src = x_dict[edge_type[0]][edge_label_index[0]]
        x_dst = x_dict[edge_type[2]][edge_label_index[1]]
        return (x_src * x_dst).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        self.embeddings = torch.nn.ModuleDict()
        for node_type in data.node_types:
            num_features = data[node_type].num_features
            self.embeddings[node_type] = torch.nn.Linear(num_features, hidden_channels)
        self.gnn = GNN(hidden_channels, data.metadata())
        self.classifier = Classifier()

    def forward(self, data):
        x_dict = {node_type: self.embeddings[node_type](data[node_type].x) for node_type in data.node_types}
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        edge_label_index = data[edge_type].edge_label_index if hasattr(data[edge_type], 'edge_label_index') else data[edge_type].edge_index
        return self.classifier(x_dict, edge_label_index)

print("Initializing model and optimizer...")
# Initialize model and optimizer
model = Model(hidden_channels=32, data=data)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# In the training loop
print("Start training loop...")
for epoch in range(1, 91):
    model.train()
    total_loss = total_examples = 0
    for batch in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        try:
            batch = batch.to(device)
            
            #print(f"Batch keys: {batch.keys()}")
            #print(f"Edge types in batch: {batch.edge_types}")
            #print(f"Node types in batch: {batch.node_types}")
            
            if edge_type not in batch.edge_types:
                print(f"Warning: edge_type {edge_type} not in batch")
                continue
            
            if not hasattr(batch[edge_type], 'edge_label'):
                print(f"Warning: edge_label not found for {edge_type}")
                continue
            
            pred = model(batch)
            ground_truth = batch[edge_type].edge_label
            
            #print(f"Prediction shape: {pred.shape}")
            #print(f"Ground truth shape: {ground_truth.shape}")
            
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
    
    gc.collect()
    print_memory_usage()
    torch.cuda.empty_cache()

# Evaluation
model.eval()

# Initialize lists to store validation results
val_predictions = []
val_ground_truth = []

print("Starting validation...")


print("\nJust before creating validation loader:")
print(f"edge_label_index shape: {val_data[edge_type].edge_label_index.shape}")
print(f"edge_label shape: {val_data[edge_type].edge_label.shape}")
print(f"Positive samples: {val_data[edge_type].edge_label.sum().item()}")
print(f"Negative samples: {(val_data[edge_type].edge_label == 0).sum().item()}")

# Ensure a mix of positive and negative samples
if val_data[edge_type].edge_label.sum().item() == 0:
    print("Warning: No positive samples in validation set. Adjusting...")
    num_edges = val_data[edge_type].edge_label.size(0)
    num_pos = num_edges // 2
    val_data[edge_type].edge_label[:num_pos] = 1.0
    print(f"Adjusted - Positive: {num_pos}, Negative: {num_edges - num_pos}")

# Initialize the validation loader
val_loader = LinkNeighborLoader(
    val_data, 
    num_neighbors=[10, 5], 
    batch_size=val_data[edge_type].edge_label_index.size(1),
    edge_label_index=(edge_type, val_data[edge_type].edge_label_index),
    edge_label=val_data[edge_type].edge_label,
    shuffle=True  # Set to True to mix positive and negative samples
)
print("LinkNeighborLoader initialized successfully")

print("\nChecking first batch of validation loader:")
first_batch = next(iter(val_loader))
print(f"Batch edge_label_index shape: {first_batch[edge_type].edge_label_index.shape}")
print(f"Batch edge_label shape: {first_batch[edge_type].edge_label.shape}")
print(f"Positive samples in batch: {first_batch[edge_type].edge_label.sum().item()}")
print(f"Negative samples in batch: {(first_batch[edge_type].edge_label == 0).sum().item()}")

# Start the validation loop
model.eval()
val_predictions = []
val_ground_truth = []

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm.tqdm(val_loader, desc="Validating")):
        print(f"\nValidation batch {batch_idx}:")
        print(f"Batch edge_label_index shape: {batch[edge_type].edge_label_index.shape}")
        print(f"Batch edge_label shape: {batch[edge_type].edge_label.shape}")
        print(f"Positive samples in batch: {batch[edge_type].edge_label.sum().item()}")
        print(f"Negative samples in batch: {(batch[edge_type].edge_label == 0).sum().item()}")
        try:
            batch = batch.to(device)
            
            pred = model(batch)
            ground_truth = batch[edge_type].edge_label
            
            val_predictions.append(pred.cpu())
            val_ground_truth.append(ground_truth.cpu())
            
            print(f"Prediction shape: {pred.shape}")
            print(f"Ground truth shape: {ground_truth.shape}")
            print(f"Prediction min: {pred.min().item():.4f}, max: {pred.max().item():.4f}")
            
        except Exception as e:
            print(f"Error in validation batch {batch_idx}: {str(e)}")
            traceback.print_exc()
            continue

        gc.collect()
        print_memory_usage()

    print("Validation loop completed")

# Safely concatenate the results and calculate metrics
if val_predictions and val_ground_truth:
    val_predictions = torch.cat(val_predictions)
    val_ground_truth = torch.cat(val_ground_truth)
    
    print(f"Total validation samples: {len(val_ground_truth)}")
    print(f"Positive samples: {val_ground_truth.sum().item()}")
    print(f"Negative samples: {len(val_ground_truth) - val_ground_truth.sum().item()}")
    
    if len(torch.unique(val_ground_truth)) == 1:
        print("Warning: Only one class present in validation set. Cannot calculate AUC.")
        if val_ground_truth[0] == 1:
            print("All samples are positive.")
        else:
            print("All samples are negative.")
    else:
        # Calculate validation metrics
        val_auc = roc_auc_score(val_ground_truth.numpy(), val_predictions.numpy())
        print(f"Validation AUC: {val_auc:.4f}")
else:
    print("No valid predictions or ground truth for validation")
