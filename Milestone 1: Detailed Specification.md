# Detailed Specification
**Dataset Collection and Creation Strategy:**

**strategies for gathering SPARQL query training data:**
* Crowdsourcing: This could involve creating a platform where community members can submit natural language queries and their corresponding SPARQL queries. To incentivize participation, we could offer compensation. However, this would require careful validation to ensure the quality of the submitted data. 
    * strategy to use DKG knowledge assets:
        * Data Extraction: Extract data from the DKG and use this data to create a variety of SPARQL queries. This could involve selecting a subset of the DKG, and writing a script to generate different SPARQL queries that extract various types of information from this subset.
        * Human Labeling: Then, we could pay people to label these queries by writing the corresponding natural language question or statement for each query.
        * Machine Labeling:  We can explore using a LLM to label queries with corresponding natural language, and use humans to validate the labels.
        * Open Source Datasets: There may already exist datasets that map natural language queries to SPARQL queries. One notable dataset is the LC-QuAD, which includes around 30,000 SPARQL queries and their corresponding natural language questions. These datasets could potentially be used as a starting point for the model, but they might not perfectly fit the context of the OriginTrail DKG, so fine tuning on the DKG will likely be a useful step.
     
**strategy for learning natural language presentation of query results:**
* Similar to training on queries, we can train the LLM to provide a natural language description of results returned from a query.  We can explore existing models, and fine tune them with examples where DKG query results are labelled with a natural language counterpart.  The labelling could be done by humans or possibly a LLM with humans in the loop to verify or tweak the results.

**strategies for interfacing with DKG knowledge assets.**

**potential open-source datasets and integration possibilities:**
* graph embeddings
  There are several popular methods for learning graph embeddings, including:
  * DeepWalk and Node2Vec: These methods work by performing random walks on the graph to generate sequences of nodes, similar to sentences in a text corpus. These sequences are then fed into a Word2Vec-like model to learn embeddings for each node. 
  * Graph Convolutional Networks (GCNs): These methods work by iteratively aggregating information from each node's neighbors to update its embedding. 
  * GraphSAGE: This method extends GCNs by allowing nodes to sample a fixed number of neighbors at each layer of the model, which can make it more scalable for large graphs. 
These are just a few examples, and the potential applications could be even broader depending on the specific characteristics of the OriginTrail Decentralized Knowledge Graph. The combination of text embeddings and graph embeddings allows you to leverage both the semantic content of the data and its structural relationships, which can provide a more holistic view of the information in the graph.  

**Architecture:** \
![Frame 1 (3)](https://github.com/DarrenZal/NatLangKG/assets/3492713/fc8fa2b4-d991-49b6-b1fa-a15550dde835) \
This diagram merges the training, testing, validation, and production processes.  
The user interface is where users can interact with the DKG in natural language.
The interface can also be where users choose DKG assets to be used to train a model or generate graph embeddings.

The NLP Engine is where the LLM and Graph embeddings are maintained.  This interfaces with the DKG to run querys.

Possible area of future research is distributing the NLP Engine such that it is not centralized, and could be securely managed by users to learn from their specific datasets.

**Sequence:** \



\
**Examples:**
* Metacrisis Mapping \
    Darren is a close collaborator with a group of people working to map the metacrisis (see Metacrisis.xyz).  This project could make use of a knwoledge graph, with nodes representing various crises (climate change, economic inequality, etc.) and edges illustrating their interconnections. This could also have nodes for solutions and their connections to the crises they address.  A natural language interface to such a map would be useful for the public to make sense of the metacrisis.  Such a metacrisis map could be used to train an AI model to facilitate natural language interaction.  The existing knowledge graph is a markdown text database which does not provide the benefits of a full semantic graph.  See https://explorer.gitcoin.co/#/round/1/0x421510312c40486965767be5ea603aa8a5707983/0x421510312c40486965767be5ea603aa8a5707983-56
* Bioregional Mapping \
    Darren is also involved with a group of people working to create a global network of bioregional economies.  This involves helping people to map their bioregions, including the resources, organizations, and people in their region.  Each bioregion could create a knowledge graph with their contextual data as a "map" which could be used to train an AI model to generate a natural language interface.  There could also be a global graph that holds knowledge relevant to all bioregions.
