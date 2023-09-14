# Technical Specification
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

**Strategies for interfacing with DKG knowledge assets.**
* Allow users to use knowledge assets to fine tune the model on their specific data.  This woud require humans in the loop, which could potentially be outsourced for a fee.
* Base level training could make use of knowledge assets, both public and private.  If a business model is developed for the models making use of such knolwedge assets, uses could be compensated for providing access to private datasets.

**Potential open-source datasets and integration possibilities:**
* graph embeddings
  There are several popular methods for learning graph embeddings, including:
  * DeepWalk and Node2Vec: These methods work by performing random walks on the graph to generate sequences of nodes, similar to sentences in a text corpus. These sequences are then fed into a Word2Vec-like model to learn embeddings for each node. 
  * Graph Convolutional Networks (GCNs): These methods work by iteratively aggregating information from each node's neighbors to update its embedding. 
  * GraphSAGE: This method extends GCNs by allowing nodes to sample a fixed number of neighbors at each layer of the model, which can make it more scalable for large graphs. 
These are just a few examples, and the potential applications could be even broader depending on the specific characteristics of the OriginTrail Decentralized Knowledge Graph. The combination of text embeddings and graph embeddings allows you to leverage both the semantic content of the data and its structural relationships, which can provide a more holistic view of the information in the graph.
* training data
  * There may already exist datasets that map natural language queries to SPARQL queries. One notable dataset is the LC-QuAD, which includes around 30,000 SPARQL queries and their corresponding natural language questions. These datasets could potentially be used as a starting point for the model, but they might not perfectly fit the context of the OriginTrail DKG, so fine tuning on the DKG will likely be a useful step.

**Architecture:** \
![Frame 1 (4)](https://github.com/DarrenZal/NatLangKG/assets/3492713/6f72a83b-0bc1-4d16-8f92-a8ca41a398b3) \
This diagram merges the training, testing, validation, and production processes.  
The user interface is where users can interact with the DKG in natural language.
The interface can also be where users choose DKG assets to be used to train a model or generate graph embeddings.
We will try to store the training data in the DKG, or in another data store if necessary.

The NLP Engine is where the LLM and Graph embeddings are maintained.  This interfaces with the DKG to run querys.

Possible area of future research is distributing the NLP Engine such that it is not centralized, and could be securely managed by users to learn from their specific datasets.

**Sequence:** \
![Frame 2 (2)](https://github.com/DarrenZal/NatLangKG/assets/3492713/37ff54db-13f9-4b42-bd4e-bf18b2122bcd) 

![Frame 3 (3)](https://github.com/DarrenZal/NatLangKG/assets/3492713/8417f43c-052f-4c38-acdd-7fc602f91fb1) 

![Frame 4](https://github.com/DarrenZal/NatLangKG/assets/3492713/11dc95c5-035e-4620-808d-8d73ab8a1b54)

