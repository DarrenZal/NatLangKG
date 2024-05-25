import re
import json
from urllib.parse import urlparse

def extract_base_url(url):
    """Extract the base URL from a full URL."""
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}/"
    else:
        return None

def infer_range_from_id(url, defined_entities, classes):
    """Infer the range class from the entity URL."""
    if url in defined_entities:
        entity_type = defined_entities[url]
        return entity_type
    else:
        for cls in classes:
            if url.startswith(cls):
                return url
        return url  # Return the URL itself if no match is found

def extract_prefixes(json_data):
    """Extract prefixes from JSON-LD data."""
    prefixes = {}
    
    def add_prefix(uri):
        prefix = uri.rsplit('#', 1)[0] + '#' if '#' in uri else uri.rsplit('/', 1)[0] + '/'
        if prefix not in prefixes.values():
            prefixes[f"p{len(prefixes)}"] = prefix

    def extract_from_value(value):
        if isinstance(value, str) and value.startswith('http'):
            add_prefix(value)
        elif isinstance(value, list):
            for item in value:
                extract_from_value(item)
        elif isinstance(value, dict):
            for k, v in value.items():
                if k.startswith('http'):
                    add_prefix(k)
                extract_from_value(v)

    if isinstance(json_data, dict):
        for k, v in json_data.items():
            if k.startswith('http'):
                add_prefix(k)
            extract_from_value(v)
    elif isinstance(json_data, list):
        for item in json_data:
            extract_from_value(item)

    return prefixes

def resolve_prefix(term, prefixes):
    """Resolve a term to its prefix and local name using the given prefixes."""
    for prefix, uri in prefixes.items():
        if term.startswith(uri):
            local_name = term[len(uri):]
            if '/' in local_name or '#' in local_name:
                if local_name.startswith(uri):
                    local_name = local_name[len(uri):]
                return prefix, local_name
            else:
                return prefix, local_name
    return None, term

def process_blank_node(blank_node, blank_node_class, classes, properties, resolved_prop, prefixes, blank_node_class_mapping):
    classes.add(blank_node_class)
    if resolved_prop:
        properties[resolved_prop]['ranges'].add(blank_node_class)
    
    # Process properties of the blank node
    for blank_node_prop, blank_node_values in blank_node.items():
        if blank_node_prop != '@id':
            prefix, local_name = resolve_prefix(blank_node_prop, prefixes)
            if prefix:
                resolved_blank_node_prop = f"{prefix}:{local_name}"
            else:
                resolved_blank_node_prop = blank_node_prop
            properties.setdefault(resolved_blank_node_prop, {'domains': set(), 'ranges': set()})
            properties[resolved_blank_node_prop]['domains'].add(blank_node_class)
            # Handle range of the blank node property based on its values
            if isinstance(blank_node_values, list):
                for blank_node_value in blank_node_values:
                    if isinstance(blank_node_value, dict):
                        if '@value' in blank_node_value:
                            literal_value = blank_node_value['@value']
                            literal_type = blank_node_value.get('@type', 'http://www.w3.org/2001/XMLSchema#string')
                            prefix, local_name = resolve_prefix(literal_type, prefixes)
                            if prefix:
                                resolved_literal_type = f"{prefix}:{local_name}"
                            else:
                                resolved_literal_type = literal_type
                            properties[resolved_blank_node_prop]['ranges'].add(resolved_literal_type)
                        elif '@id' in blank_node_value:
                            entity_url = blank_node_value['@id']
                            if entity_url.startswith('_:'):  # Blank node
                                if entity_url in blank_node_class_mapping:
                                    mapped_class = blank_node_class_mapping[entity_url]
                                    properties[resolved_blank_node_prop]['ranges'].add(mapped_class)
                                else:
                                    properties[resolved_blank_node_prop]['ranges'].add(blank_node_class)
                            else:
                                # Resolve the entity URL to a prefix:local_name format
                                prefix, local_name = resolve_prefix(entity_url, prefixes)
                                if prefix:
                                    resolved_entity = f"{prefix}:{local_name}"
                                else:
                                    resolved_entity = entity_url
                                properties[resolved_blank_node_prop]['ranges'].add(resolved_entity)
            else:
                if isinstance(blank_node_values, dict):
                    if '@value' in blank_node_values:
                        literal_value = blank_node_values['@value']
                        literal_type = blank_node_values.get('@type', 'http://www.w3.org/2001/XMLSchema#string')
                        prefix, local_name = resolve_prefix(literal_type, prefixes)
                        if prefix:
                            resolved_literal_type = f"{prefix}:{local_name}"
                        else:
                            resolved_literal_type = literal_type
                        properties[resolved_blank_node_prop]['ranges'].add(resolved_literal_type)
                    elif '@id' in blank_node_values:
                        entity_url = blank_node_values['@id']
                        prefix, local_name = resolve_prefix(entity_url, prefixes)
                        if prefix:
                            resolved_entity = f"{prefix}:{local_name}"
                        else:
                            resolved_entity = entity_url
                        properties[resolved_blank_node_prop]['ranges'].add(resolved_entity)

def convert_to_hashable(value):
    if isinstance(value, dict):
        return tuple(sorted((k, convert_to_hashable(v)) for k, v in value.items()))
    elif isinstance(value, list):
        return tuple(convert_to_hashable(v) for v in value)
    else:
        return value

def process_jsonld_data(json_data, defined_entities, all_prefixes, classes, properties, ontology_comments):
    prefixes = extract_prefixes(json_data)
    all_prefixes.update(prefixes)

    assertions = json_data  # Assertion data is already extracted

    if isinstance(assertions, dict):
        assertions = [assertions]

    # Collect blank node properties
    blank_node_properties = {}
    for item in assertions:
        if item.get('@id', '').startswith('_:'):
            blank_node_properties[item['@id']] = item

    # Process entities and populate defined_entities dictionary
    for item in assertions:
        entity_id = item.get('@id')
        entity_types = item.get('@type', [])
        if not isinstance(entity_types, list):
            entity_types = [entity_types]
        if entity_types:
            for entity_type in entity_types:
                prefix, local_name = resolve_prefix(entity_type, prefixes)
                if prefix:
                    defined_entities[entity_id] = f"{prefix}:{local_name}"
                else:
                    defined_entities[entity_id] = entity_type
                classes.add(entity_type)
    
    # Process properties
    blank_node_class_mapping = {}
    attribute_class_mapping = {}
    for item in assertions:
        entity_id = item.get('@id')
        entity_types = item.get('@type', [])
        if not isinstance(entity_types, list):
            entity_types = [entity_types]

        for prop, values in item.items():
            if prop.startswith('@'):  # Skip JSON-LD reserved properties
                continue
            prefix, local_name = resolve_prefix(prop, prefixes)
            if prefix:
                resolved_prop = f"{prefix}:{local_name}"
            else:
                resolved_prop = prop
            properties.setdefault(resolved_prop, {'domains': set(), 'ranges': set(), 'comments': set()})
            if resolved_prop in ontology_comments:
                properties[resolved_prop]['comments'].add(ontology_comments[resolved_prop])
            if entity_types:
                for type_uri in entity_types:
                    prefix, local_name = resolve_prefix(type_uri, prefixes)
                    if prefix:
                        properties[resolved_prop]['domains'].add(f"{prefix}:{local_name}")
                    else:
                        properties[resolved_prop]['domains'].add(type_uri)

            if isinstance(values, list):
                for value in values:
                    if isinstance(value, dict):
                        if '@id' in value:
                            entity_url = value['@id']
                            if entity_url.startswith('_:'):  # Blank node
                                blank_node_class = blank_node_class_mapping.get(entity_url)
                                if not blank_node_class:
                                    blank_node_props = blank_node_properties[entity_url]
                                    blank_node_attrs = convert_to_hashable(blank_node_props)
                                    if blank_node_attrs in attribute_class_mapping:
                                        blank_node_class = attribute_class_mapping[blank_node_attrs]
                                    else:
                                        blank_node_class = f"BlankNode_Class{entity_url}"
                                        attribute_class_mapping[blank_node_attrs] = blank_node_class
                                    blank_node_class_mapping[entity_url] = blank_node_class
                                blank_node_props = blank_node_properties.get(entity_url, {})
                                process_blank_node(blank_node_props, blank_node_class, classes, properties, resolved_prop, prefixes, blank_node_class_mapping)
                            else:
                                range_class = infer_range_from_id(entity_url, defined_entities, classes)
                                if range_class != "owl:Thing":
                                    properties[resolved_prop]['ranges'].add(range_class)
                        elif '@value' in value:
                            if isinstance(value['@value'], str) and value['@value'].startswith('http'):
                                entity_url = value['@value']
                                if entity_url not in defined_entities:
                                    entity_type = None
                                    for assertion in assertions:
                                        if assertion.get('@id') == entity_url:
                                            entity_type = assertion.get('@type')
                                            if isinstance(entity_type, list):
                                                entity_type = entity_type[0]  # Take the first type from the list
                                            break
                                    if entity_type:
                                        prefix, local_name = resolve_prefix(entity_type, prefixes)
                                        if prefix:
                                            defined_entities[entity_url] = f"{prefix}:{local_name}"
                                        else:
                                            defined_entities[entity_url] = entity_type
                                    else:
                                        defined_entities[entity_url] = "owl:Thing"
                                range_class = infer_range_from_id(entity_url, defined_entities, classes)
                                properties[resolved_prop]['ranges'].add(range_class)
                    elif isinstance(value, str):
                        if value.startswith('http'):
                            range_class = infer_range_from_id(value, defined_entities, classes)
                            prefix, local_name = resolve_prefix(range_class, prefixes)
                            if prefix:
                                resolved_range_class = f"{prefix}:{local_name}"
                            else:
                                resolved_range_class = range_class
                            properties[resolved_prop]['ranges'].add(resolved_range_class)
                        else:
                            properties[resolved_prop]['ranges'].add(value)
            elif isinstance(values, dict):
                if '@id' in values:
                    entity_url = values['@id']
                    if entity_url.startswith('_:'):  # Blank node
                        blank_node_class = blank_node_class_mapping.get(entity_url)
                        if not blank_node_class:
                            blank_node_props = blank_node_properties[entity_url]
                            blank_node_attrs = convert_to_hashable(blank_node_props)
                            if blank_node_attrs in attribute_class_mapping:
                                blank_node_class = attribute_class_mapping[blank_node_attrs]
                            else:
                                blank_node_class = f"<BlankNode_Class{entity_url}>"
                                attribute_class_mapping[blank_node_attrs] = blank_node_class
                            blank_node_class_mapping[entity_url] = blank_node_class
                        process_blank_node(values, blank_node_class, classes, properties, resolved_prop, prefixes, blank_node_class_mapping)
                    else:
                        range_class = infer_range_from_id(entity_url, defined_entities, classes)
                        prefix, local_name = resolve_prefix(range_class, prefixes)
                        if prefix:
                            resolved_range_class = f"{prefix}:{local_name}"
                        else:
                            resolved_range_class = range_class
                        properties[resolved_prop]['ranges'].add(resolved_range_class)
                elif '@value' in values:
                    if isinstance(values['@value'], str) and values['@value'].startswith('http'):
                        range_class = infer_range_from_id(values['@value'], defined_entities)
                        prefix, local_name = resolve_prefix(range_class, prefixes)
                        if prefix:
                            properties[resolved_prop]['ranges'].add(f"{prefix}:{local_name}")
                        else:
                            properties[resolved_prop]['ranges'].add(range_class)
                    else:
                        literal_type = values.get('@type', 'http://www.w3.org/2001/XMLSchema#string')
                        prefix, local_name = resolve_prefix(literal_type, prefixes)
                        if prefix:
                            properties[resolved_prop]['ranges'].add(f"{prefix}:{local_name}")
                        else:
                            properties[resolved_prop]['ranges'].add(literal_type)
            elif isinstance(values, str):
                if values.startswith('http'):
                    range_class = infer_range_from_id(values, defined_entities, classes)
                    prefix, local_name = resolve_prefix(range_class, prefixes)
                    if prefix:
                        resolved_range_class = f"{prefix}:{local_name}"
                    else:
                        resolved_range_class = range_class
                    properties[resolved_prop]['ranges'].add(resolved_range_class)
                else:
                    properties[resolved_prop]['ranges'].add(values)
                    
def consolidate_classes(properties):
    class_properties = {}
    for prop, info in properties.items():
        for cls in info['domains']:
            if cls.startswith("BlankNode_Class"):
                class_properties.setdefault(cls, []).append((prop, 'domain'))
        for cls in info['ranges']:
            if cls.startswith("BlankNode_Class"):
                class_properties.setdefault(cls, []).append((prop, 'range'))

    reverse_mapping = {}
    consolidated_classes = set()  # Set to keep track of all consolidated classes
    for cls, props in class_properties.items():
        props = frozenset(props)  # Make it hashable
        reverse_mapping.setdefault(props, []).append(cls)
        if len(reverse_mapping[props]) > 1:
            consolidated_classes.update(reverse_mapping[props])

    consolidation = {}
    for classes in reverse_mapping.values():
        sorted_classes = sorted(classes)
        representative = sorted_classes[0]  # Pick the first class when sorted as representative
        for cls in classes:
            consolidation[cls] = representative

    return consolidation, consolidated_classes

def apply_consolidation(properties, consolidation):
    # Apply the consolidation mapping to properties
    for prop, info in properties.items():
        new_domains = {consolidation.get(cls, cls) for cls in info['domains']}
        new_ranges = {consolidation.get(cls, cls) for cls in info['ranges']}
        properties[prop] = {'domains': new_domains, 'ranges': new_ranges, 'comments': info['comments']}

def generate_ontology(jsonld_data_list, ontology_comments):
    classes = set()
    properties = {}
    defined_entities = {}  # Keep track of defined entities and their types
    all_prefixes = {}

    # Process JSON-LD data
    for json_data in jsonld_data_list:
        prefixes = extract_prefixes(json_data)
        all_prefixes.update(prefixes)
        process_jsonld_data(json_data, defined_entities, all_prefixes, classes, properties, ontology_comments)

    consolidation, consolidated_classes = consolidate_classes(properties)
    apply_consolidation(properties, consolidation)

    # Writing the ontology with prefixes
    with open('Ontology.ttl', 'w') as file:
        file.write("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n")
        file.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n")

        # Determine the base URL
        base_url = None
        for prefix, uri in all_prefixes.items():
            if uri.endswith('#'):
                base_url = uri[:-1]
                break

        if base_url:
            file.write(f"@base <{base_url}> .\n")

        for prefix, uri in all_prefixes.items():
            file.write(f"@prefix {prefix}: <{uri}> .\n")
        file.write("\n")

        for class_type in classes:
            if class_type not in consolidated_classes or class_type in consolidation.values():
                prefix, local_name = resolve_prefix(class_type, all_prefixes)
                if prefix:
                    file.write(f"{prefix}:{local_name} a owl:Class .\n")
                else:
                    file.write(f"<{class_type}> a owl:Class .\n")

        file.write("\n")

        for prop_uri, prop_info in properties.items():
            prefix, local_name = resolve_prefix(prop_uri, all_prefixes)
            if prefix:
                prop_write = f"{prefix}:{local_name}"
            else:
                prop_write = f"<{prop_uri}>"

            file.write(f"{prop_write} a owl:ObjectProperty ;\n")

            # Handle comments
            if prop_info['comments']:
                for comment in prop_info['comments']:
                    file.write(f"    rdfs:comment \"{comment}\" ;\n")

            # Handle domains
            if prop_info['domains']:
                domain_classes = []
                for domain in prop_info['domains']:
                    prefix, local_name = resolve_prefix(domain, all_prefixes)
                    if prefix:
                        domain_classes.append(f"{prefix}:{local_name}")
                    else:
                        domain_classes.append(f"<{domain}>")
                domain_classes_str = ' '.join(domain_classes)
                file.write(f"    rdfs:domain [ a owl:Class ; owl:unionOf ({domain_classes_str}) ] ;\n")

            # Handle ranges
            if prop_info['ranges']:
                range_classes = set()
                for range_class in prop_info['ranges']:
                    prefix, local_name = resolve_prefix(range_class, all_prefixes)
                    if prefix:
                        range_classes.add(f"{prefix}:{local_name}")
                    else:
                        range_classes.add(f"<{range_class}>")
                range_classes_str = ' '.join(range_classes)
                file.write(f"    rdfs:range [ a owl:Class ; owl:unionOf ({range_classes_str}) ] .\n")
            else:
                file.write("    rdfs:range owl:Thing .\n")

            file.write("\n")
