import json
import requests
from rdflib import Graph, URIRef, BNode, RDF, RDFS, Literal, OWL
from .TTLtoJSON import ttl_to_json
from rdflib import Namespace
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
        current_bnode = next_bnode
    return first_bnode

def inline_union_list(ontology_graph, used_ontology_graph, subject, predicate, blank_node, processed_lists, actual_domains):
    union_of_uri = URIRef("http://www.w3.org/2002/07/owl#unionOf")
    first_uri = RDF.first
    rest_uri = RDF.rest
    nil_uri = RDF.nil
    if (subject, predicate) in processed_lists:
        return  # Skip if already processed

    if str(predicate) == 'http://www.w3.org/2000/01/rdf-schema#domain' and str(subject) in actual_domains:
        return

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

def generate_ontology_contextLookup(jsonld_data_list):
    ontology_urls = {}
    for dataset_data in jsonld_data_list:
        assertions = dataset_data.get("assertion", []) if isinstance(dataset_data, dict) else dataset_data
        for item in assertions:
            for predicate in item:
                if predicate.startswith("@") or predicate.startswith("_:"):
                    continue
                if predicate.startswith("http://schema.org/"):
                    ontology_url = "https://schema.org/version/latest/schemaorg-current-https.jsonld"
                    ontology_urls[ontology_url] = ontology_urls.get(ontology_url, 0) + 1
                else:
                    ontology_url = predicate.rsplit('#', 1)[0]
                    if not ontology_url.endswith('/'):
                        ontology_url += '/'
                    ontology_urls[ontology_url] = ontology_urls.get(ontology_url, 0) + 1

    if not ontology_urls:
        raise ValueError("No ontology URLs found in the dataset.")

    main_ontology_url = max(ontology_urls, key=ontology_urls.get)
    print("main_ontology_url")
    print(main_ontology_url)

    ontology_data = None
    for ontology_url in ontology_urls:
        try:
            if ontology_url.startswith("https://schema.org/"):
                response = requests.get(ontology_url)
                ontology_data = response.json()
            else:
                ontology_url_no_slash = ontology_url.rstrip('/')
                response = requests.get(ontology_url_no_slash, headers={'Accept': 'text/turtle, application/rdf+xml, application/ld+json'}, allow_redirects=True)
                content_type = response.headers.get('Content-Type')
                if 'text/turtle' in content_type or 'text/plain' in content_type:
                    ontology_data_ttl = response.text
                    ontology_data = ttl_to_json(ontology_data_ttl)
                elif 'application/rdf+xml' in content_type:
                    ontology_data_rdf = response.text
                    g = Graph()
                    g.parse(data=ontology_data_rdf, format='xml')
                    ontology_data = json.loads(g.serialize(format='json-ld'))
                elif 'application/ld+json' in content_type or 'application/json' in content_type:
                    ontology_data = response.json()
                else:
                    print(f"Unsupported Content-Type: {content_type} for ontology URL: {ontology_url}")
                    continue

            if ontology_url == main_ontology_url:
                break
        except Exception as e:
            print(f"Error fetching ontology URL: {ontology_url}. Error: {str(e)}")

    if ontology_data is None:
        raise ValueError("Failed to fetch the main ontology.")

    ontology_graph = Graph()
    ontology_graph.parse(data=json.dumps(ontology_data), format="json-ld")
    
    actual_domains = {}
    actual_ranges = {}
    processed_lists = {}
    used_ontology_graph = Graph()
    
    for dataset_data in jsonld_data_list:
        assertions = dataset_data.get("assertion", []) if isinstance(dataset_data, dict) else dataset_data
        for item in assertions:
            subject_type = item.get('@type', [None])[0]
            for predicate, objects in item.items():
                if predicate.startswith("@"):
                    continue
                predicate_uri = URIRef(predicate)
                for obj in objects:
                    if subject_type:
                        actual_domains.setdefault(str(predicate_uri), set()).add(str(subject_type))
                    if isinstance(obj, dict):
                        if '@id' in obj:
                            object_uri = obj['@id']
                            if object_uri.startswith('_:'):
                                continue
                            object_type = obj.get('@type', [None])[0]
                            if object_type:
                                actual_ranges.setdefault(str(predicate_uri), set()).add(str(object_type))
                        elif '@value' in obj:
                            actual_ranges.setdefault(str(predicate_uri), set()).add("http://www.w3.org/2001/XMLSchema#string")
                if (predicate_uri, None, None) in ontology_graph:
                    for s, p, o in ontology_graph.triples((predicate_uri, None, None)):
                        if isinstance(o, BNode):
                            inline_union_list(ontology_graph, used_ontology_graph, s, p, o, processed_lists, actual_domains)
                        else:
                            used_ontology_graph.add((s, p, o))

    # Now handle updating domains and ranges based on actual usage
    for predicate, domains in actual_domains.items():
        adjust_domain_range(used_ontology_graph, URIRef(predicate), domains, 'domain')
    for predicate, ranges in actual_ranges.items():
        adjust_domain_range(used_ontology_graph, URIRef(predicate), ranges, 'range')

    # Collect used prefixes
    used_prefixes = set()
    for s, p, o in used_ontology_graph:
        for uri in [s, p, o]:
            if isinstance(uri, URIRef):
                namespace, _ = str(uri).rsplit('#', 1) if '#' in str(uri) else str(uri).rsplit('/', 1)
                used_prefixes.add(namespace)

    # Generate prefixes dynamically from the ontology data
    prefixes = {}
    for key, value in ontology_data.get('@context', {}).items():
        if isinstance(value, str) and any(value.startswith(ns) for ns in used_prefixes):
            if '#' in value:
                prefix, namespace = value.rsplit('#', 1)
                prefixes[key] = prefix + '#'
            elif '/' in value:
                prefix, namespace = value.rsplit('/', 1)
                prefixes[key] = prefix + '/'
    
    # Create a new context with the generated prefixes
    context = {"@context": prefixes}

    # Bind the prefixes to the graph
    for prefix, namespace in prefixes.items():
        used_ontology_graph.bind(prefix, Namespace(namespace))

    # Serialize the used ontology graph with the updated context
    used_ontology_data = used_ontology_graph.serialize(format='json-ld', context=context["@context"], indent=2)
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
