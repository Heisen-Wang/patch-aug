
module load Stages/2020  GCCcore/.10.3.0
module load PyTorch/1.8.1-Python-3.8.5
module load torchvision/0.9.1-Python-3.8.5
module load tqdm/4.62.3-Python-3.8.5

source venv_dataloader/bin/activate
python dataloader_downstream.py
