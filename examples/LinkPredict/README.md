# Link Prediction on the DKG

## Install Requirements:

```bash
conda create -n PyG python=3.9   10?
conda activate PyG
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cpuonly -c pytorch
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-geometric
pip install ipykernel
python -m ipykernel install --user --name=PyG
pip install python-dotenv
pip install dkg
