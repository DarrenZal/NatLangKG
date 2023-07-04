# NatLangKG
**AI-Enhanced Semantic Interfacing for Decentralized Knowledge Graphs
**

**Description:**
Our project aims to harness the capabilities of advanced language models to create a natural language interface for the OriginTrail Decentralized Knowledge Graph (DKG). We plan to empower non-technical users to query Knowledge Assets with ease, transforming natural language input into SPARQL and providing understandable natural language outputs. Our initiative promotes a broader adoption of the DKG by making it more user-friendly, while also contributing to the democratization and trustworthiness of AI by rooting the model in a verifiable data source.

To facilitate integration with other projects and maximize synergy, our project will utilize existing transformer-based text embeddings and, as a novel approach, leverage graph embeddings for a better understanding of the DKG structure. We will provide compensation to contributors for valuable contributions such as training data and model improvements.

**Milestones, Timeline, and Expected Impact:
**
**Phase 1 (Month 1): Dataset Collection and Creation
**The first phase involves creating a dataset of SPARQL queries matched with their natural language counterparts. We'll encourage community contributions, verified through a validation process and rewarded through smart contracts. During this phase, we will also explore and identify parts of the DKG to be used for creating graph embeddings[1]. 

strategies for gathering SPARQL query training data:
* Crowdsourcing: This could involve creating a platform where community members can submit natural language queries and their corresponding SPARQL queries. To incentivize participation, we could offer compensation. However, this would require careful validation to ensure the quality of the submitted data.
    * strategy to DKG knowledge assets:
        * Data Extraction: Extract data from the DKG and use this data to create a variety of SPARQL queries. This could involve selecting a subset of the DKG, and writing a script to generate different SPARQL queries that extract various types of information from this subset.
        * Human Labeling: Then, we could pay people to label these queries by writing the corresponding natural language question or statement for each query.
* Open Source Datasets: There may already exist datasets that map natural language queries to SPARQL queries. One notable dataset is the LC-QuAD, which includes around 30,000 SPARQL queries and their corresponding natural language questions. These datasets could potentially be used as a starting point for your model, but they might not perfectly fit the context of the OriginTrail DKG. 
* Synthetic Data Generation: We could potentially generate synthetic training data. This could involve creating a wide range of SPARQL queries, executing them against your graph, and then using the results to generate natural language questions. However, this requires careful design to ensure that the synthetic data accurately represents the queries users will actually submit. 
* Data Augmentation: This is a technique where existing data is modified to create new examples. For example, one could take a single question-query pair and generate many similar pairs by replacing synonyms, rephrasing the question, or altering the query structure without changing its meaning.  Speaking of synonyms, it may be helpful to add alternative labels to nodes and edges to improve the performance of machine learning models trained on the data.  Synonyms could be added to the graph, enriching knowledge assets, however this additional data would need quality control and using it would increase the complexity and computational resources of training the model.  Perhaps an area of further research.

**Phase 2 (Months 2-4): Development of Graph Embeddings and LLM Training
**Building on the insights gained from the initial dataset collection, we'll develop graph embeddings to complement the existing text embeddings, which will provide a richer understanding of the DKG and improve the precision of the natural language to SPARQL conversion. Concurrently, we will fine-tune an open-source Large Language Model using the dataset from Phase 1.

**Graph Embeddings:
**There are several popular methods for learning graph embeddings, including:
* DeepWalk and Node2Vec: These methods work by performing random walks on the graph to generate sequences of nodes, similar to sentences in a text corpus. These sequences are then fed into a Word2Vec-like model to learn embeddings for each node. 
* Graph Convolutional Networks (GCNs): These methods work by iteratively aggregating information from each node's neighbors to update its embedding. 
* GraphSAGE: This method extends GCNs by allowing nodes to sample a fixed number of neighbors at each layer of the model, which can make it more scalable for large graphs. 
These are just a few examples, and the potential applications could be even broader depending on the specific characteristics of the OriginTrail Decentralized Knowledge Graph. The combination of text embeddings and graph embeddings allows you to leverage both the semantic content of the data and its structural relationships, which can provide a more holistic view of the information in the graph.  Thi

**LLM Training
**We will use the dataset from Phase 1 to fine-tune a pre-trained open source Large Language model to convert between natural language and SPARQL, as well as giving a natural language response back with the query results.  We will research other options as well such as using models geared specifically to do language-to-logic transformations.


**Phase 3 (Month 5): Integration
**We will blend our freshly trained LLM with the graph embeddings to construct a comprehensive tool that captures the best of both worlds. The goal is to leverage the strength of the language model in understanding natural language and the power of the graph embeddings in understanding the structure and context of the DKG.

**Phase 4 (Month 6): Launch, Beta Testing, and Refinement
**With the tool now ready, we will wrap it in a user-friendly interface and conduct a comprehensive beta testing phase, rewarding testers for their time and invaluable feedback. Insights from the testing phase will be instrumental in refining the AI model and interface, ensuring it is optimized for end-users.

**Expected Impact:**
By transforming how users interact with the OriginTrail DKG, we expect to boost its utility and accessibility, promoting its widespread use. By enriching knowledge assets with structural metadata, our solution will provide a more nuanced understanding of the DKG, making the data querying process more intuitive, and lowering barriers for non-technical users.

**Budget:**
We request a budget of $20K-$30K to fairly compensate contributors, beta testers, and our development team. This budget will also cover the costs of cloud computing resources for model training and software development costs. Our goal is to utilize these funds in a way that aligns with the decentralized and community-driven spirit of the OriginTrail DKG.

As the lead developer, I am open to the idea of working collaboratively with other developers and sharing the workload. This collaboration would further foster the spirit of community and decentralization that underpins this project.


**Appendix:**
[1] 1. graph embeddings and text embeddings can both be extremely useful for a variety of tasks, especially when used together. Here are a few ways that these types of embeddings could create value:
    * Semantic Search: This is likely one of the most immediate applications. By combining text embeddings with graph embeddings, you can create a more robust semantic search engine for the knowledge graph. Text embeddings can be used to understand the semantic content of the user's query, while graph embeddings can help provide results that are not only semantically relevant but also structurally relevant within the context of the graph. 
    * Knowledge Graph Completion: As mentioned earlier, graph embeddings can be used to predict missing links in the knowledge graph. Combining this with text embeddings could potentially make this task more accurate. For example, the text embeddings could provide additional context that helps the model make better predictions about whether a link should exist between two nodes. 
    * Entity Resolution: Graph embeddings can be used to identify entities in the knowledge graph that are similar to each other, which can be useful for entity resolution tasks. Combining this with text embeddings could potentially make the resolution process more accurate, as the text embeddings could help identify semantic similarities between entities that might not be immediately apparent from the structure of the graph alone.
    * Recommendation Systems: In a system where users can own and trade knowledge assets, graph embeddings and text embeddings could be used together to build a more effective recommendation system. The system could use text embeddings to understand the semantic content of the assets and graph embeddings to understand their positions within the graph, allowing it to make recommendations that are both semantically relevant to the user's interests and structurally important within the context of the graph.
    * Anomaly Detection: Graph embeddings can be useful for anomaly detection in the knowledge graph. Anomalies in the graph could indicate erroneous data or potential fraud, and graph embeddings could help identify nodes or subgraphs that are structurally unusual. Combined with text embeddings, this could provide a powerful tool for ensuring the integrity of the data in the knowledge graph.
