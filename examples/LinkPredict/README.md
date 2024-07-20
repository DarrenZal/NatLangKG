# Link Prediction on the DKG

## Install Requirements:

```bash
# Create and activate the conda environment
conda create -n PyG python=3.9   10?
conda activate PyG

# Install PyTorch and related packages for CPU
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cpuonly -c pytorch

# Install torch-geometric dependencies
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-geometric

# Install other necessary packages
pip install ipykernel
pip install python-dotenv
pip install dkg
pip install scikit-learn
pip install tqdm

# Install the Jupyter kernel
python -m ipykernel install --user --name=PyG
```

## Usage

To train the GNN to predict links for a set of knowledge asset, for a given pink type, run the following command:

```bash
python main.py <KA_1> <KA_2> ... <KA_N> <link_type>
```
- `<KA_DID_1>`, `<KA_DID_2>`, ..., `<KA_DID_N>`: The list of Knowledge Asset DIDs to be processed.
- `<link_type>` The link type to train for prediction.  should be in the format '("subjectType", "LinkType", "ObjectType")'

Example:
```
python main.py did:dkg:otp:2043/0x5cac41237127f94c2d21dae0b14bfefa99880630/7695243 '("InvestmentOrGrant", "investee", "Organization")' 
```
