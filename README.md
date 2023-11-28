# NatLangKG
**AI-Enhanced Semantic Interfacing for Decentralized Knowledge Graphs**

**Description:**
Our project aims to harness the capabilities of advanced language models to create a natural language interface for the OriginTrail Decentralized Knowledge Graph (DKG). We plan to empower non-technical users to query Knowledge Assets with ease, transforming natural language input into SPARQL and providing understandable natural language outputs. Our initiative promotes a broader adoption of the DKG by making it more user-friendly, while also contributing to the democratization and trustworthiness of AI by rooting the model in a verifiable data source.

To facilitate integration with other projects and maximize synergy, our project will utilize existing transformer-based text embeddings. We will provide compensation to contributors for valuable contributions such as training data and model improvements.

**Milestones, Timeline, and Expected Impact:** \
**Milestone 1 (Completion by September 15th): Detailed Specification** \
Dataset Collection and Creation Strategy:
   * Define strategies for gathering SPARQL query training data.
   * Specify strategies for interfacing with DKG knowledge assets.
   * Review potential open-source datasets and integration possibilities.
   * Architectural and sequence diagrams for the proposed system.

**Milestone 2 (Completion by the end of October): Prototype Development**
Development of Question Answering system:
   * Based on the insights from Milestone 1, initiate the development of necessary components, such as text and graph embeddings[1], for entity and relation mapping.
   * Train a Large Language Model with datasets of natural language-and-SPARQL questions.
     update: this proved to be expensive for training large datasets such as LC-QuAD for many epochs, especcially considering that new models like GPT-4 are already good at creating SPARQL queries given the relevant ontology and entities.  We plan to explore Few shot learning with a few examples as a more affordable option.
   * Prepare a prototype that can be demonstrated, which involves the conversion between natural language and SPARQL and providing natural language outputs.
     update: can be found here: https://github.com/DarrenZal/ChatDKG/tree/main/examples/langchain

Integration:
   * Blend the trained LLM with the developed text and graph embeddings to create a preliminary tool that combines both.
   * Ensure the prototype is hosted in an externally accessible environment.

**Milestone 3 (Completion by December 31st 2023): Finalization and Launch**
Refinement and Optimization:
   * Refine the AI model based on insights and feedback from the prototype phase.
   * Optimize the user interface and backend processes.
   * Launch, Beta Testing, and Final Refinement:

Release the tool with a user-friendly interface.
   * Conduct comprehensive beta testing, collating feedback and further optimizing the model.
   * Final documentation including Milestone 1 specification.
   * Package final codebase for delivery.

**Expected Impact:**
By transforming how users interact with the OriginTrail DKG, we expect to boost its utility and accessibility, promoting its widespread use. By enriching knowledge assets with structural metadata, our solution will provide a more nuanced understanding of the DKG, making the data querying process more intuitive, and lowering barriers for non-technical users.

**Architecture:** \
![Frame 1 (3)](https://github.com/DarrenZal/NatLangKG/assets/3492713/fc8fa2b4-d991-49b6-b1fa-a15550dde835) \
This diagram merges the training, testing, validation, and production processes.  
The user interface is where users can interact with the DKG in natural language.
The interface can also be where users choose DKG assets to be used to train a model or generate graph embeddings.

The NLP Engine is where the LLM and Graph embeddings are maintained.  This interfaces with the DKG to run querys.

Possible area of future research is distributing the NLP Engine such that it is not centralized, and could be securely managed by users to learn from their specific datasets.

\
**Examples:**
* Metacrisis Mapping \
    Darren is a close collaborator with a group of people working to map the metacrisis (see Metacrisis.xyz).  This project could make use of a knwoledge graph, with nodes representing various crises (climate change, economic inequality, etc.) and edges illustrating their interconnections. This could also have nodes for solutions and their connections to the crises they address.  A natural language interface to such a map would be useful for the public to make sense of the metacrisis.  Such a metacrisis map could be used to train an AI model to facilitate natural language interaction.  The existing knowledge graph is a markdown text database which does not provide the benefits of a full semantic graph.  See https://explorer.gitcoin.co/#/round/1/0x421510312c40486965767be5ea603aa8a5707983/0x421510312c40486965767be5ea603aa8a5707983-56
* Bioregional Mapping \
    Darren is also involved with a group of people working to create a global network of bioregional economies.  This involves helping people to map their bioregions, including the resources, organizations, and people in their region.  Each bioregion could create a knowledge graph with their contextual data as a "map" which could be used to train an AI model to generate a natural language interface.  There could also be a global graph that holds knowledge relevant to all bioregions.

\
**Appendix:**
[1] graph embeddings and text embeddings can both be extremely useful for a variety of tasks, especially when used together. Here are a few ways that these types of embeddings could create value:
   * Semantic Search: This is likely one of the most immediate applications. By combining text embeddings with graph embeddings, you can create a more robust semantic search engine for the knowledge graph. Text embeddings can be used to understand the semantic content of the user's query, while graph embeddings can help provide results that are not only semantically relevant but also structurally relevant within the context of the graph.
   * Knowledge Graph Completion: As mentioned earlier, graph embeddings can be used to predict missing links in the knowledge graph. Combining this with text embeddings could potentially make this task more accurate. For example, the text embeddings could provide additional context that helps the model make better predictions about whether a link should exist between two nodes.
   * Entity Resolution: Graph embeddings can be used to identify entities in the knowledge graph that are similar to each other, which can be useful for entity resolution tasks. Combining this with text embeddings could potentially make the resolution process more accurate, as the text embeddings could help identify semantic similarities between entities that might not be immediately apparent from the structure of the graph alone.
   * Recommendation Systems: In a system where users can own and trade knowledge assets, graph embeddings and text embeddings could be used together to build a more effective recommendation system. The system could use text embeddings to understand the semantic content of the assets and graph embeddings to understand their positions within the graph, allowing it to make recommendations that are both semantically relevant to the user's interests and structurally important within the context of the graph.
   * Anomaly Detection: Graph embeddings can be useful for anomaly detection in the knowledge graph. Anomalies in the graph could indicate erroneous data or potential fraud, and graph embeddings could help identify nodes or subgraphs that are structurally unusual. Combined with text embeddings, this could provide a powerful tool for ensuring the integrity of the data in the knowledge graph.

There are several popular methods for learning graph embeddings, including:
* DeepWalk and Node2Vec: These methods work by performing random walks on the graph to generate sequences of nodes, similar to sentences in a text corpus. These sequences are then fed into a Word2Vec-like model to learn embeddings for each node. 
* Graph Convolutional Networks (GCNs): These methods work by iteratively aggregating information from each node's neighbors to update its embedding. 
* GraphSAGE: This method extends GCNs by allowing nodes to sample a fixed number of neighbors at each layer of the model, which can make it more scalable for large graphs. 
These are just a few examples, and the potential applications could be even broader depending on the specific characteristics of the OriginTrail Decentralized Knowledge Graph. The combination of text embeddings and graph embeddings allows you to leverage both the semantic content of the data and its structural relationships, which can provide a more holistic view of the information in the graph.  

Questions for future research:  
  A question-answering system typically takes in a question posed in natural language, processes and understands the question, retrieves the relevant information, and delivers the answer, usually also in natural language.
Translating a natural language question to a SPARQL query, querying a knowledge graph, and then translating the results back into natural language is one approach to building such a system. However, there are other, potentially more complex, aspects of question answering that could be considered.
Here are a few:
* Handling Ambiguity: Given the inherent ambiguity in natural language, enhancing your system to better handle this ambiguity could be a valuable improvement. This could involve techniques for entity disambiguation (i.e., determining which specific entity in the graph a given name or description refers to) and for resolving other kinds of linguistic ambiguities. While there can still be significant complexity here, there are existing techniques and tools that can be applied to this task.
    * Entity Recognition and Disambiguation: When processing natural language queries, the first step is to identify the named entities mentioned in the query. Named Entity Recognition (NER) models can be used for this task. These models identify and categorize named entities in text into predefined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc. However, a named entity could refer to multiple things in the real world. For instance, "Apple" could refer to the fruit, the tech company, or even a record company. This is where Entity Disambiguation comes in. Given the context, you would need to figure out which specific entity in the knowledge graph the term refers to. This is where text embeddings can be especially useful, as they can help to capture the context in which an entity is mentioned.
    * Property and Relation Disambiguation: In a similar vein, property and relation names could be ambiguous in natural language. For example, if a user asks "Who created Batman?", they could be asking about the individual writer or the company that owns the rights to the character. Here, again, context is crucial. You might use a combination of heuristics based on your knowledge of the graph structure (e.g., what kinds of entities are connected by the "created" relation) and machine learning techniques to disambiguate.
    * Referential Ambiguity: This refers to situations where it's not clear what a pronoun or other referential expression is referring to. For example, in the sentence "John told Bill that he failed the exam", it's not clear whether "he" refers to John or Bill. Solving this requires a process called co-reference resolution. There are NLP tools and libraries that can help with this task, although it can be quite complex.
    * Graph Embeddings: Graph embeddings can be used to disambiguate entities, properties, and relations based on their positions in the graph. For example, if two entities have similar embeddings, they are likely to be similar in some way or have similar properties, and this information can be used to disambiguate between 
    * Overall, handling ambiguity in natural language queries requires a combination of NLP and machine learning techniques, a deep understanding of your knowledge graph, and possibly also some manual rules or heuristics based on your specific use case. While it can be quite complex, it's a crucial aspect of making a natural language interface to a knowledge graph that's robust and useful in practice. 
* Multi-step Reasoning: Depending on the complexity of your graph and the nature of the questions you want to support, adding multi-step reasoning capabilities could provide significant value. This could involve chaining together multiple SPARQL queries, or using a more complex reasoning approach. The complexity here can vary significantly depending on the specifics. 
* Inference: Implementing basic inferential capabilities could also provide significant value, allowing your system to answer questions that can't be directly answered based on the information in the graph. Depending on the nature of your graph and the kinds of inferences you want to support, this could involve adding rules-based logic, probabilistic reasoning, or even machine learning-based inference. 
* Temporal and Spatial Reasoning: This can be more complex as it requires not just understanding these concepts but also having the data structured in such a way that allows these kinds of queries. However, if your graph includes a significant amount of temporal or spatial information, and if these kinds of queries are important for your use case, this could provide significant value. 
* Explanations: Providing explanations for how the system arrived at its answers can be complex, particularly if it involves generating natural language explanations. However, it can greatly increase the transparency and trustworthiness of the system. If your users need to understand the reasoning behind the system's answers, this could provide significant value. 
Each of these aspects adds additional complexity to the task of building a question-answering system, but they can also greatly increase the system's usefulness and capabilities.
