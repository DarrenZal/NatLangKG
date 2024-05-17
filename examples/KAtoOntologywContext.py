import json
import requests
from rdflib import Graph, URIRef, BNode, RDF, RDFS, Literal, OWL

def create_rdf_list(union_list, graph):
    if not union_list:
        return RDF.nil
    current_bnode = BNode()
    first_bnode = current_bnode
    for i, item in enumerate(union_list):
        next_bnode = BNode() if i < len(union_list) - 1 else RDF.nil
        graph.add((current_bnode, RDF.first, URIRef(item)))
        graph.add((current_bnode, RDF.rest, next_bnode))
        print(f"Adding list item: {item} with rest: {next_bnode}")
        current_bnode = next_bnode
    return first_bnode

def inline_union_list(ontology_graph, used_ontology_graph, subject, predicate, blank_node, processed_lists, actual_domains):
    union_of_uri = URIRef("http://www.w3.org/2002/07/owl#unionOf")
    first_uri = RDF.first
    rest_uri = RDF.rest
    nil_uri = RDF.nil
    if (subject, predicate) in processed_lists:
        return  # Skip if already processed

    print(f"Processing union list for {subject} {predicate}")
    if str(predicate) == 'http://www.w3.org/2000/01/rdf-schema#domain' and str(subject) in actual_domains:
        print(f"Skipping domain union list for {subject} as it is already inferred from the dataset")
        return

    print(f"Actual domains inside inline_union_list: {actual_domains}")  # Debugging statement

    current_list_node = ontology_graph.value(blank_node, union_of_uri)
    union_list = []
    while current_list_node != nil_uri:
        first_element = ontology_graph.value(current_list_node, first_uri)
        if first_element:
            union_list.append(str(first_element))
        current_list_node = ontology_graph.value(current_list_node, rest_uri)
    rdf_list = create_rdf_list(union_list, used_ontology_graph)
    used_ontology_graph.set((subject, predicate, rdf_list))
    processed_lists[(subject, predicate)] = True

def adjust_domain_range(graph, uri, new_values, property_type):
    prop_uri = RDFS.domain if property_type == 'domain' else RDFS.range
    for s, p, o in list(graph.triples((uri, prop_uri, None))):
        graph.remove((s, p, o))
    if new_values:
        if len(new_values) == 1:
            graph.add((uri, prop_uri, URIRef(next(iter(new_values)))))
        else:
            union_bnode = create_rdf_list(list(new_values), graph)
            graph.add((uri, prop_uri, union_bnode))
    print(f"Updated {property_type} for {uri}: {new_values}")

def extract_used_ontology(dataset_files, ontology_url, output_file):
    response = requests.get(ontology_url)
    ontology_data = response.json()
    ontology_graph = Graph()
    ontology_graph.parse(data=json.dumps(ontology_data), format="json-ld")
    used_ontology_graph = Graph()
    processed_lists = {}
    actual_domains = {}
    actual_ranges = {}
    
    for dataset_file in dataset_files:
        with open(dataset_file, "r") as file:
            dataset_data = json.load(file)
        assertions = dataset_data.get("assertion", []) if isinstance(dataset_data, dict) else dataset_data
        for item in assertions:
            subject_type = item.get('@type', [None])[0]
            print(f"Processing subject: {item['@id']} with type: {subject_type}")
            for predicate, objects in item.items():
                if predicate.startswith("@"):
                    continue
                predicate_uri = URIRef(predicate)
                print(f"Processing predicate: {predicate_uri}")
                for obj in objects:
                    print(f"Processing object: {obj}")
                    if subject_type:
                        print(f"Adding domain: {subject_type} for predicate: {predicate_uri}")
                        actual_domains.setdefault(str(predicate_uri), set()).add(str(subject_type))
                    if isinstance(obj, dict):
                        print(f"Object keys: {obj.keys()}")
                        if '@id' in obj:
                            object_uri = obj['@id']
                            if object_uri.startswith('_:'):
                                continue
                            object_type = obj.get('@type', [None])[0]
                            print(f"Processing object: {object_uri} with type: {object_type}")
                            if object_type:
                                print(f"Adding range: {object_type} for predicate: {predicate_uri}")
                                actual_ranges.setdefault(str(predicate_uri), set()).add(str(object_type))
                        elif '@value' in obj:
                            print(f"Adding range: xsd:string for predicate: {predicate_uri}")
                            actual_ranges.setdefault(str(predicate_uri), set()).add("http://www.w3.org/2001/XMLSchema#string")
                        else:
                            print(f"Skipping object: {obj}")
                    else:
                        print(f"Skipping non-dictionary object: {obj}")
                print(f"Actual domains after processing predicate {predicate_uri}: {actual_domains}")
                print(f"Actual ranges after processing predicate {predicate_uri}: {actual_ranges}")
                if (predicate_uri, None, None) in ontology_graph:
                    print(f"Predicate found in ontology: {predicate_uri}")
                    for s, p, o in ontology_graph.triples((predicate_uri, None, None)):
                        if isinstance(o, BNode):
                            print(f"Calling inline_union_list with actual_domains: {actual_domains}")
                            inline_union_list(ontology_graph, used_ontology_graph, s, p, o, processed_lists, actual_domains)
                        else:
                            used_ontology_graph.add((s, p, o))

    print("Actual domains found in the dataset:")
    for predicate, domains in actual_domains.items():
        print(f"{predicate}: {domains}")

    print("Actual ranges found in the dataset:")
    for predicate, ranges in actual_ranges.items():
        print(f"{predicate}: {ranges}")

    # Now handle updating domains and ranges based on actual usage
    for predicate, domains in actual_domains.items():
        adjust_domain_range(used_ontology_graph, URIRef(predicate), domains, 'domain')
    for predicate, ranges in actual_ranges.items():
        adjust_domain_range(used_ontology_graph, URIRef(predicate), ranges, 'range')

    used_ontology_data = used_ontology_graph.serialize(format='json-ld', context=ontology_data['@context'], indent=2)
    with open(output_file, "w") as file:
        file.write(used_ontology_data)

jsonld_files = ['./KnowledgeAsset.json']
output_file_path = './ontology.jsonld'
ontology_url = 'https://darrenzal.github.io/ChatDKG/REAcontext.jsonld'
extract_used_ontology(jsonld_files, ontology_url, output_file_path)
