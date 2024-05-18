import rdflib
import json

def ttl_to_json(ttl_data):
    # Load the Turtle data into an RDFlib graph
    g = rdflib.Graph()
    g.parse(data=ttl_data, format="turtle")

    # Extract prefixes from the Turtle data
    prefixes = {prefix: str(namespace) for prefix, namespace in g.namespaces()}

    # Serialize the graph to JSON-LD using the generated context
    jsonld_data = g.serialize(format='json-ld', context=prefixes, indent=4)
    jsonld_object = json.loads(jsonld_data)

    # Optionally, ensure that the output is wrapped under a single key if it helps with further processing
    wrapped_jsonld = {"@context": prefixes, "@graph": jsonld_object} if isinstance(jsonld_object, list) else {"@context": prefixes, **jsonld_object}

    return wrapped_jsonld
