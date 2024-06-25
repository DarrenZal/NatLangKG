import argparse
import json
import torch
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from rdflib import Graph, URIRef, RDF, RDFS, Literal, BNode, XSD, OWL
from sklearn.model_selection import train_test_split

from rdflib import URIRef, Literal

def load_jsonld(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    g = Graph().parse(data=json.dumps(data), format='json-ld')
    
    node_dict = {}
    edge_index = []
    node_features = []
    
    def get_node_id(node):
        if isinstance(node, URIRef):
            return str(node)
        elif isinstance(node, Literal):
            return f"Literal({node.n3()})"
        elif isinstance(node, BNode):
            return f"BNode({node})"
        else:
            return str(node)
    
    for s, p, o in g:
        s_id = get_node_id(s)
        p_id = get_node_id(p)
        o_id = get_node_id(o)
        
        if s_id not in node_dict:
            node_dict[s_id] = len(node_dict)
            node_features.append([1, 0, 0])  # [is_subject, is_predicate, is_object]
        if o_id not in node_dict:
            node_dict[o_id] = len(node_dict)
            node_features.append([0, 0, 1])
        if p_id not in node_dict:
            node_dict[p_id] = len(node_dict)
            node_features.append([0, 1, 0])
        
        edge_index.append([node_dict[s_id], node_dict[o_id]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    # Create reverse dictionary for node lookup
    node_dict_reverse = {v: k for k, v in node_dict.items()}
    
    data = Data(x=node_features, edge_index=edge_index, num_nodes=len(node_dict))
    data.node_dict = node_dict
    data.node_dict_reverse = node_dict_reverse
    
    return data

def load_ontology(file_path):
    with open(file_path, 'r') as f:
        ontology_data = json.load(f)
    
    g = Graph().parse(data=json.dumps(ontology_data), format='json-ld')
    return g

def is_valid_link(ontology, subject, predicate, object, verbose=False):
    if verbose:
        print(f"Checking validity of link: {subject} -> {predicate} -> {object}")
    
    def get_types(entity):
        if isinstance(entity, Literal):
            return [entity.datatype] if entity.datatype else [XSD.string]
        elif isinstance(entity, BNode):
            return [RDFS.Resource]
        elif isinstance(entity, str):
            if entity.startswith('http://') or entity.startswith('https://'):
                types = list(ontology.objects(URIRef(entity), RDF.type))
                return types if types else [OWL.Thing]
            else:
                return [XSD.string]
        else:
            types = list(ontology.objects(URIRef(str(entity)), RDF.type))
            return types if types else [OWL.Thing]
    
    subject_types = get_types(subject)
    predicate_types = get_types(predicate)
    object_types = get_types(object)
    
    if verbose:
        print(f"Subject types: {subject_types}")
        print(f"Predicate types: {predicate_types}")
        print(f"Object types: {object_types}")
    
    # Check if the predicate is a valid property in the ontology
    if any(isinstance(p, URIRef) for p in predicate_types):
        for p in predicate_types:
            if (p, RDF.type, RDF.Property) in ontology:
                return True
    
    # If we couldn't validate based on the ontology, consider it valid if all parts are defined
    subject_in_ontology = any(t for t in subject_types if not isinstance(t, Literal))
    predicate_in_ontology = any(t for t in predicate_types if not isinstance(t, Literal))
    object_in_ontology = any(t for t in object_types if not isinstance(t, Literal))
    
    if subject_in_ontology and predicate_in_ontology and object_in_ontology:
        if verbose:
            print("All parts of the triple are defined in the ontology")
        return True
    
    if verbose:
        print("Link is not valid according to the ontology")
    return False

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.triple_pred = torch.nn.Linear(out_channels * 3, 1)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return self.triple_pred(torch.cat([z[edge_label_index[0]], z[edge_label_index[1]], z[edge_label_index[2]]], dim=-1)).squeeze()

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

def train_model(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    
    # Use all edges for training
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    
    # Create positive triples
    pos_triples = torch.zeros((3, num_edges), dtype=torch.long)
    pos_triples[0] = edge_index[0]  # subject
    pos_triples[1] = torch.arange(num_edges)  # predicate (using edge index as a placeholder)
    pos_triples[2] = edge_index[1]  # object
    
    # Create negative triples
    neg_edge_index = torch_geometric.utils.negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=num_edges,
    )
    neg_triples = torch.zeros((3, num_edges), dtype=torch.long)
    neg_triples[0] = neg_edge_index[0]  # subject
    neg_triples[1] = torch.arange(num_edges)  # predicate (using edge index as a placeholder)
    neg_triples[2] = neg_edge_index[1]  # object

    pos_out = model(data.x, edge_index, pos_triples)
    neg_out = model(data.x, edge_index, neg_triples)

    loss = criterion(pos_out, torch.ones_like(pos_out))
    loss += criterion(neg_out, torch.zeros_like(neg_out))

    loss.backward()
    optimizer.step()
    
    return loss.item()

def parse_node(node_str):
    if node_str.startswith('Literal('):
        # Extract the content of the Literal
        content = node_str[8:-1]  # Remove 'Literal(' and ')'
        return Literal(content)
    elif node_str.startswith('BNode('):
        # Extract the BNode identifier
        identifier = node_str[6:-1]  # Remove 'BNode(' and ')'
        return BNode(identifier)
    elif node_str.startswith('N') and len(node_str) == 32:
        # This is likely an internal node identifier, return as is
        return node_str
    else:
        # For all other cases, return the string as is
        return node_str

def predict_links(model, data, ontology, top_k=10, max_triples=1000000):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        
        all_triples = [(s, p, o) for s in range(data.num_nodes) 
                                 for p in range(data.num_nodes) 
                                 for o in range(data.num_nodes) 
                                 if s != o]
        
        # Randomly sample triples if there are too many
        if len(all_triples) > max_triples:
            all_triples = random.sample(all_triples, max_triples)
        
        edge_label_index = torch.tensor(all_triples, dtype=torch.long).t().contiguous()
        scores = model.decode(z, edge_label_index)
        
        existing_edges = set(map(tuple, data.edge_index.t().tolist()))
        predicted_edges = []
        for (s, p, o), score in zip(all_triples, scores):
            if (s, o) not in existing_edges:
                subject = data.node_dict_reverse[s]
                predicate = data.node_dict_reverse[p]
                object = data.node_dict_reverse[o]
                
                # Skip internal node IDs
                if any(isinstance(node, str) and node.startswith('N') and len(node) == 32 for node in [subject, predicate, object]):
                    continue
                
                # Parse the node strings
                subject = parse_node_string(subject)
                predicate = parse_node_string(predicate)
                object = parse_node_string(object)
                
                valid = is_valid_link(ontology, subject, predicate, object)
                
                # Adjust score based on validity
                adjusted_score = score.item() * (1.5 if valid else 0.5)
                
                predicted_edges.append((subject, predicate, object, adjusted_score, valid))
        
        return sorted(predicted_edges, key=lambda x: x[3], reverse=True)[:top_k]

def parse_node_string(node_str):
    if node_str.startswith('Literal('):
        return Literal(eval(node_str[8:-1]))  # Remove 'Literal(' and ')', then eval
    elif node_str.startswith('BNode('):
        return BNode(node_str[6:-1])  # Remove 'BNode(' and ')'
    else:
        return node_str

def main(json_ld_file, ontology_file):
    data = load_jsonld(json_ld_file)
    ontology = load_ontology(ontology_file)
    
    model = LinkPredictor(in_channels=data.num_features, hidden_channels=64, out_channels=32)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        loss = train_model(model, data, optimizer, criterion)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

    # Predict new links
    predicted_links = predict_links(model, data, ontology)
    
    valid_count = sum(1 for _, _, _, _, valid in predicted_links if valid)
    
    print(f"\nPredicted {len(predicted_links)} links, {valid_count} valid.")
    print("\nTop 10 predicted links:")
    for subject, predicate, object, score, valid in predicted_links:
        subject_str = f'"{subject}"' if isinstance(subject, Literal) else str(subject)
        predicate_str = f'"{predicate}"' if isinstance(predicate, Literal) else str(predicate)
        object_str = f'"{object}"' if isinstance(object, Literal) else str(object)
        print(f"{subject_str} -[{predicate_str}]-> {object_str}: Score {score:.4f}, Valid: {valid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JSON-LD Link Prediction')
    parser.add_argument('json_ld_file', type=str, help='Path to the JSON-LD file')
    parser.add_argument('ontology_file', type=str, help='Path to the ontology JSON-LD file')
    args = parser.parse_args()
    
    main(args.json_ld_file, args.ontology_file)
