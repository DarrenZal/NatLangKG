import argparse
import json
import os
import asyncio
from dkg import DKG
from dkg.providers import BlockchainProvider, NodeHTTPProvider
from dkg.constants import BLOCKCHAINS
from dotenv import load_dotenv
from .KAtoOntologywContext import generate_ontology_with_context
from .KAtoOntology import generate_ontology
from rdflib import Graph

load_dotenv()

# Initialize the DKG client on OriginTrail DKG Testnet
ot_node_hostname = os.getenv("OT_NODE_HOSTNAME_MAINNET")+":8900"
node_provider = NodeHTTPProvider(ot_node_hostname)
blockchain_provider = BlockchainProvider(
    "mainnet",
    "otp:2043",
    os.getenv("RPC_ENDPOINT_MAINNET"),
    os.getenv("WALLET_PRIVATE_KEY")
)

# Initialize the DKG client
dkg = DKG(node_provider, blockchain_provider)


def get_asset_data(ual):
    try:
        get_asset_result = dkg.asset.get(ual)
        # Extract and convert the assertion from TTL to JSON-LD
        if 'public' in get_asset_result and 'assertion' in get_asset_result['public']:
            ttl_data = get_asset_result['public']['assertion']
            g = Graph()
            g.parse(data=ttl_data, format='ttl')
            jsonld_data = g.serialize(format='json-ld', indent=2)
            # Return only the assertion part as JSON-LD
            return json.loads(jsonld_data)
        else:
            print(f"No assertion data found for {ual}")
            return None
    except Exception as e:
        print(f"Error getting asset data for {ual}: {e}")
        return None

async def load_kas_and_generate_ontology(ka_dids, ontology_url=None):
    ka_data_list = []
    for i, did in enumerate(ka_dids):
        ka_data = get_asset_data(did)
        if ka_data:
            ka_data_list.append(ka_data)
            # Write each ka_data to its own file
            ka_filename = f"ka_data_{i}.json"
            with open(ka_filename, 'w') as ka_file:
                json.dump(ka_data, ka_file, indent=2)

    # Assuming generate_ontology accepts JSON-LD data directly
    if ontology_url:
        generate_ontology_with_context(ka_data_list, ontology_url)
    else:
        generate_ontology(ka_data_list)

    print(f"Ontology file generated: Ontology.ttl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load KAs and generate ontology")
    parser.add_argument("ka_dids", nargs="+", help="List of KA DIDs")
    parser.add_argument("--ontology-url", help="Ontology URL")
    args = parser.parse_args()

    # Use asyncio to run the main async function
    asyncio.run(load_kas_and_generate_ontology(args.ka_dids, args.ontology_url))
