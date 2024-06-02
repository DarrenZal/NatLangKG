import os
import json
import asyncio
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from rdflib import Graph
from Utils.loadKAs import load_kas_and_generate_ontology

load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

def read_ontology_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def execute_sparql_query(graph, query):
    try:
        # Execute the SPARQL query on the loaded graph
        results = graph.query(query)
        return results
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
                        "Include all necessary namespace prefixes in the SPARQL query. "
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

async def main(prompt, ka_dids, ontology_url=None):
    # Always re-fetch KA data and regenerate the ontology
    graph = await load_kas_and_generate_ontology(ka_dids, ontology_url)

    # Read the ontology content for generating SPARQL query
    ontology_content = read_ontology_file("Ontology.ttl")
    print("Ontology Content:")
    print(ontology_content)

    # Generate the SPARQL query
    generated_query = generate_sparql_query(prompt, ontology_content)
    print("Generated SPARQL query:")
    print(generated_query)

    # Execute the generated SPARQL query
    query_results = execute_sparql_query(graph, generated_query)
    print("Query results:")
    for row in query_results:
        print(row)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate and execute SPARQL queries based on a given prompt.")
    parser.add_argument("prompt", type=str, help="The natural language prompt to generate the SPARQL query.")
    parser.add_argument("ka_dids", nargs='+', help="List of KA DIDs.")
    parser.add_argument("--ontology-url", type=str, help="Ontology URL", default=None)

    # Parse arguments
    args = parser.parse_args()

    # Run the async main function
    asyncio.run(main(args.prompt, args.ka_dids, args.ontology_url))
