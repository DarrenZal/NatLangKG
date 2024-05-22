import argparse
import json
import os
from dkg import DKG
from dotenv import load_dotenv
from Utils.KAtoOntologywContext import generate_ontology_with_context
from Utils.KAtoOntology import generate_ontology

load_dotenv()

# Initialize the DKG client on OriginTrail DKG Testnet
ot_node_hostname = os.getenv("OT_NODE_HOSTNAME")+":8900"
node_provider = NodeHTTPProvider(ot_node_hostname)
blockchain_provider = BlockchainProvider(
    os.getenv("RPC_ENDPOINT"), 
    os.getenv("WALLET_PRIVATE_KEY")
)

# Initialize the DKG client
dkg = DKG(node_provider, blockchain_provider)

async def get_asset_data(ual):
    try:
        get_asset_result = await dkg.asset.get(ual)
        return get_asset_result
    except Exception as e:
        print(f"Error getting asset data for {ual}: {e}")
        return None

async def load_kas_and_generate_ontology(ka_dids, output_file, ontology_url=None):
    ka_data_list = []
    for did in ka_dids:
        ka_data = await get_asset_data(did)
        if ka_data:
            ka_data_list.append(ka_data)

    if ontology_url:
        ontology = generate_ontology_with_context(ka_data_list, ontology_url, output_file)
    else:
        ontology = generate_ontology(ka_data_list, output_file)

    print(f"Ontology file generated: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load KAs and generate ontology")
    parser.add_argument("ka_dids", nargs="+", help="List of KA DIDs")
    parser.add_argument("--output-file", default="ontology.ttl", help="Output file path for the generated ontology")
    parser.add_argument("--ontology-url", help="Ontology URL")
    args = parser.parse_args()

    load_kas_and_generate_ontology(args.ka_dids, args.output_file, args.ontology_url)
