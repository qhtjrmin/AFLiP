# AFLiP
AFLiP: An Access–Recomputation Decoupling Framework for Efficient and Structure-Preserving GNN-based Link Prediction

## Tested Environment
The experiments were conducted under the following environment:
- Ubuntu 20.04.2 LTS (64-bit)
- Python 3.10
- PyTorch 1.13.0
- CUDA Toolkit 11.6 with NVIDIA Driver 510

Additional library dependencies can be found in `environment.yml`.

## Getting started
#### Clone the repository
```bash
$ cd ~/
$ git clone https://github.com/qhtjrmin/AFLiP.git
$ cd AFLiP
```
#### Create the conda environment:
```
$ conda env create -f environment.yml
$ conda activate aflip
```
#### Example of running code:
```
$ python run_aflip.py --num_runs 5 --epochs 100 --train_mode "aflip" --model SAGE --batch_size 2048 --device 0 --dataset collab
```
Additional source code components and detailed running guidelines will be released before publication.

## License
This project is licensed under the [MIT License](./LICENSE).
