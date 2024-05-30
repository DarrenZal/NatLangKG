import os
import json
import asyncio
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from dkg import DKG
from dkg.providers import BlockchainProvider, NodeHTTPProvider
from Utils.loadKAs import load_kas_and_generate_ontology

load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

def read_ontology_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def execute_sparql_query(query):
    try:
        # Initialize the DKG client on OriginTrail DKG Testnet
        ot_node_hostname = os.getenv("OT_NODE_HOSTNAME_MAINNET")+":8900"
        node_provider = NodeHTTPProvider(ot_node_hostname)
        blockchain_provider = BlockchainProvider(
            "mainnet",
            "otp:2043",
            os.getenv("RPC_ENDPOINT_MAINNET"), 
            os.getenv("WALLET_PRIVATE_KEY_MAINNET"),
        )

        # Initialize the DKG client
        dkg = DKG(node_provider, blockchain_provider)

        # Execute the query
        query_graph_result = dkg.graph.query(query, repository="privateCurrent")

        if query_graph_result:
            return query_graph_result
        else:
            return []

    except Exception as e:
        print(f"Error during SPARQL query execution: {e}")
        return []

def generate_sparql_query(prompt, ontology_content):
    try:
        # Call the OpenAI ChatCompletion API
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that generates SPARQL queries based on natural language prompts and a provided ontology. "
                        "Your task is to create a SPARQL query that retrieves the relevant information from the knowledge graph to answer the given prompt. "
                        "Use the provided ontology to understand the structure and relationships of the data. "
                        "Ensure that the generated query respects the object properties' domain and range defined in the ontology. "
                        "If the prompt requires aggregation, make sure to aggregate the total values before applying any filters. "
                        "Include details like IDs or names in the query results for clarity. "
                        "Use 'DISTINCT' in the select when possible to avoid duplicate results. "
                        "Try to retrieve all relevant attributes for the subject in question when possible. "
                        "The response should only contain the generated SPARQL query, without any additional text or explanation."
                    )
                },
                {
                    "role": "user",
                    "content": f"Prompt: {prompt}\nOntology: {ontology_content}"
                }
            ]
        )

        generated_query = completion.choices[0].message.content.strip()
        # Ensure no extra characters or quotes are present
        generated_query = generated_query.replace("```sparql", "").replace("```", "").strip()
        return generated_query

    except Exception as e:
        print(f"Error occurred during SPARQL query generation: {e}")
        return ""

async def main(prompt, ontology_url=None):
    ontology_file_path = "Ontology.ttl"

    # Check if the ontology file exists
    if not os.path.exists(ontology_file_path):
        print("Ontology file not found. Please input a list of KA DIDs to generate the ontology.")
        ka_dids = input("Enter KA DIDs separated by spaces: ").split()
        ontology_url = input("Enter Ontology URL (or press Enter to skip): ").strip() or None
        await load_kas_and_generate_ontology(ka_dids, ontology_url)

    # Check again to ensure the ontology file was created
    if os.path.exists(ontology_file_path):
        ontology_content = read_ontology_file(ontology_file_path)

        # Generate the SPARQL query
        generated_query = generate_sparql_query(prompt, ontology_content)
        print("Generated SPARQL query:")
        print(generated_query)

        # Execute the generated SPARQL query
        query_results = execute_sparql_query(generated_query)
        print("Query results:")
        print(json.dumps(query_results, indent=2))
    else:
        print("Failed to generate ontology file. Aborting.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate and execute SPARQL queries based on a given prompt.")
    parser.add_argument("prompt", type=str, help="The natural language prompt to generate the SPARQL query.")

    # Parse arguments
    args = parser.parse_args()

    # Run the async main function
    asyncio.run(main(args.prompt))
