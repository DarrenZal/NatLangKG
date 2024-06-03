# AI-Powered SPARQL Query Generator and Executor

This repository contains Python scripts that work together to enable natural language querying of Knowledge Assets (KAs) stored in the OriginTrail Decentralized Knowledge Graph (DKG) using AI-generated SPARQL queries.

## Components

1. `GenSPARQL.py`: This script takes in a natual language prompt along with a list of Knowledge Assets, and an optional ontology url.  It queries the DKG for the input Knowledge Assets, loads them into a graph db, generates an Ontology from the Knowledge Assets, and prompts an LLM to generate a SPARQL query for the natural language prompt and ontology. The generated query is then executed to retrieve the relevant information, which is returned to the user.

## Pre-requisites

- Python 3.10 or higher.
- Access to an OriginTrail DKG node. You can setup your own by following instructions [here](https://docs.origintrail.io/decentralized-knowledge-graph-layer-2/testnet-node-setup-instructions/setup-instructions-dockerless).  Note, you will need to whitelist the IP of the machine running this code so your node will accept requests, instructions [here](https://docs.origintrail.io/decentralized-knowledge-graph/node-setup-instructions/useful-resources/manually-configuring-your-node).
- OpenAI API key

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```

2. Navigate to main directory:
   ```bash
   cd NatLangKG/examples/NatLangSPARQL
   ```   

3. Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

4. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables

You'll need to setup your environment variables. Copy the .env.example to a new .env file:

```bash
cp .env.example .env
```

Open the .env file and replace the placeholders with your actual values. The file should look like this:

```makefile
OT_NODE_HOSTNAME=<Your OT Node Hostname>
WALLET_PUBLIC_KEY=<Your Wallet Public Key>
WALLET_PRIVATE_KEY=<Your Wallet Private Key>
OPENAI_KEY=<Your OpenAI API Key>
RPC_ENDPOINT=<Your Blockchain RPC URL>
```

## Usage

### GenSPARQL.py

To generate a SPARQL query and execute it on the OriginTrail DKG, run the following command:

```bash
python GenSPARQL.py "<NATURAL_LANGUAGE_PROMPT>" <KA_DID_1> <KA_DID_2> ... <KA_DID_N> [--ontology-url <ONTOLOGY_URL>]
```

- `"<NATURAL_LANGUAGE_PROMPT>"`: The natural language prompt describing the information you want to retrieve from the KAs.
- `<KA_DID_1>`, `<KA_DID_2>`, ..., `<KA_DID_N>`: The list of Knowledge Asset DIDs to be processed.
- `--ontology-url <ONTOLOGY_URL>` (optional): The URL that returns a JSON-LD Ontology representing the entire ontology used by the data. If provided, the script will extract the subset of the ontology used by the specified KAs.

Example:
```
python GenSPARQL.py "Find the total accounting value of all economic resources" did:dkg:otp:2043/0x5cac41237127f94c2d21dae0b14bfefa99880630/6632018
```


## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
