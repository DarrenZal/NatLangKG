import logging
import json
from rdflib import Graph

logger = logging.getLogger(__name__)

def jsonld_to_rdf(jsonld_obj, filename):
    g = Graph().parse(data=json.dumps(jsonld_obj), format='json-ld')
    turtle_filename = f'{filename}.ttl'
    g.serialize(destination=turtle_filename, format='turtle')
    logger.info(f'File successfully converted to {turtle_filename}')
    return turtle_filename

def convert_jsonld_to_ttl(jsonld_filename):
    try:
        with open(jsonld_filename) as json_file:
            jsonld_obj = json.load(json_file)
        turtle_filename = jsonld_to_rdf(jsonld_obj, jsonld_filename.split('.json')[0])
        logger.info(f'File successfully converted to {turtle_filename}')
        return turtle_filename
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    import getopt
    from sys import argv

    FORMAT = '%(message)s'
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)

    def help():
        logger.info('Usage: python converter.py --file=doc.json')

    try:
        opts, args = getopt.getopt(argv[1:], "", ["file="])
    except getopt.GetoptError as e:
        logger.error(f"Error parsing options: {e}")
        help()
        exit(2)

    jsonld_filename = ''

    for opt, arg in opts:
        if opt == '--file':
            jsonld_filename = arg

    if jsonld_filename == '':
        help()
        exit(2)

    convert_jsonld_to_ttl(jsonld_filename)
