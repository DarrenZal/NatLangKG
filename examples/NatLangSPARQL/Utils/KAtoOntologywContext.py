import json
import requests
from rdflib import Graph, URIRef, BNode, RDF, RDFS, Literal, OWL
from .JSONLDtoTTL import convert_jsonld_to_ttl

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

def generate_ontology_with_context(jsonld_data_list, ontology_url):
    response = requests.get(ontology_url)
    if response.status_code == 200:
        try:
            ontology_data = response.json()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {ontology_url}: {e}")
            print(f"Response content: {response.text}")
            return
    else:
        print(f"Failed to fetch ontology from {ontology_url}. Status code: {response.status_code}")
        print(f"Response content: {response.text}")
        return

    ontology_graph = Graph()
    ontology_graph.parse(data=json.dumps(ontology_data), format="json-ld")
    used_ontology_graph = Graph()
    processed_lists = {}
    actual_domains = {}
    actual_ranges = {}

    for dataset_data in jsonld_data_list:
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

    # Now handle updating domains and ranges based on actual usage
    for predicate, domains in actual_domains.items():
        adjust_domain_range(used_ontology_graph, URIRef(predicate), domains, 'domain')
    for predicate, ranges in actual_ranges.items():
        adjust_domain_range(used_ontology_graph, URIRef(predicate), ranges, 'range')

    used_ontology_data = used_ontology_graph.serialize(format='json-ld', context=ontology_data['@context'], indent=2)
    output_file = "Ontology.json"
    with open(output_file, "w") as file:
        file.write(used_ontology_data)

    # Convert the JSON-LD file to Turtle format
    convert_jsonld_to_ttl(output_file)

    # Validate dataset against the ontology
    validate_dataset_against_ontology(jsonld_data_list, used_ontology_graph)

def validate_dataset_against_ontology(jsonld_data_list, ontology_graph):
    def get_valid_domains(ontology_graph, predicate_uri):
        domains = set()
        for domain in ontology_graph.objects(predicate_uri, RDFS.domain):
            if isinstance(domain, BNode):
                print(f"Processing domain BNode: {domain} for predicate: {predicate_uri}")
                current_list_node = ontology_graph.value(domain, OWL.unionOf)
                if current_list_node:
                    while current_list_node and current_list_node != RDF.nil:
                        union_item = ontology_graph.value(current_list_node, RDF.first)
                        if union_item:
                            domains.add(str(union_item))
                        current_list_node = ontology_graph.value(current_list_node, RDF.rest)
                else:
                    for item in ontology_graph.items(domain):
                        domains.add(str(item))
            else:
                domains.add(str(domain))
        return domains

    def get_valid_ranges(ontology_graph, predicate_uri):
        ranges = set()
        for range_ in ontology_graph.objects(predicate_uri, RDFS.range):
            if isinstance(range_, BNode):
                print(f"Processing range BNode: {range_} for predicate: {predicate_uri}")
                current_list_node = ontology_graph.value(range_, OWL.unionOf)
                if current_list_node:
                    while current_list_node and current_list_node != RDF.nil:
                        union_item = ontology_graph.value(current_list_node, RDF.first)
                        if union_item:
                            ranges.add(str(union_item))
                        current_list_node = ontology_graph.value(current_list_node, RDF.rest)
                else:
                    for item in ontology_graph.items(range_):
                        ranges.add(str(item))
            else:
                ranges.add(str(range_))
        return ranges

    for dataset_data in jsonld_data_list:
        assertions = dataset_data.get("assertion", []) if isinstance(dataset_data, dict) else dataset_data
        for item in assertions:
            subject_type = item.get('@type', [None])[0]
            print(f"Validating subject: {item.get('@id')} with type: {subject_type}")
            for predicate, objects in item.items():
                if predicate.startswith("@"):
                    continue
                predicate_uri = URIRef(predicate)
                valid_domains = get_valid_domains(ontology_graph, predicate_uri)
                valid_ranges = get_valid_ranges(ontology_graph, predicate_uri)
                print(f"Valid domains for predicate '{predicate}': {valid_domains}")
                print(f"Valid ranges for predicate '{predicate}': {valid_ranges}")
                # Validate subject type against the ontology
                if subject_type:
                    if str(subject_type) not in valid_domains:
                        raise ValueError(f"Invalid subject type '{subject_type}' for predicate '{predicate}'")
                for obj in objects:
                    if isinstance(obj, dict):
                        if '@id' in obj:
                            object_type = obj.get('@type', [None])[0]
                            if object_type:
                                if str(object_type) not in valid_ranges:
                                    raise ValueError(f"Invalid object type '{object_type}' for predicate '{predicate}'")
                        elif '@value' in obj:
                            if "http://www.w3.org/2001/XMLSchema#string" not in valid_ranges:
                                raise ValueError(f"Invalid literal type for predicate '{predicate}'")
    print("")
    print("Dataset validated")
    print("")
