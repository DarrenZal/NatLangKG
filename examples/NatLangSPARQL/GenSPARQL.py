import os
import json
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from dkg import DKG

load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

def read_ontology_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def execute_sparql_query(query):
    try:
        # Initialize DKG
        ot_node_hostname = os.getenv("OT_NODE_HOSTNAME") + ":8900"
        dkg = DKG(
            endpoint=ot_node_hostname,
            port=8900,
            blockchain={
                "name": "otp::testnet",
                "publicKey": os.getenv("WALLET_PUBLIC_KEY"),
                "privateKey": os.getenv("WALLET_PRIVATE_KEY"),
            },
        )

        # Execute the query
        query_graph_result = dkg.graph.query(query, repository="privateCurrent")
        print("query_graph_result: ", query_graph_result)

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
            model="gpt-4-0125-preview",
            temperature=0.1,
            response_format={"type": "json_object"},
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
        return generated_query

    except Exception as e:
        print(f"Error occurred during SPARQL query generation: {e}")
        return ""

async def main(prompt):
    ontology_file_path = "ontology.json"
    ontology_content = read_ontology_file(ontology_file_path)

    # Generate the SPARQL query
    generated_query = generate_sparql_query(prompt, ontology_content)
    print("Generated SPARQL query:")
    print(generated_query)

    # Execute the generated SPARQL query
    query_results = execute_sparql_query(generated_query)
    print("Query results:")
    print(json.dumps(query_results, indent=2))

if __name__ == "__main__":
    import sys

    # Get the prompt from command line argument
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Enter your prompt here"

    # Run the async main function
    asyncio.run(main(prompt))
