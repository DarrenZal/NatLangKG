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
