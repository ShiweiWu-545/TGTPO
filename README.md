# TGTPO

![image](https://github.com/ShiweiWu-545/TGTPO/blob/main/data/overview.jpg)

This repository contains the deep learning model introduced in the paper "Learning protein affinity codes with dynamic residues selection and multidimensional feature interactions using graph transformer". It predicts changes in binding energy upon mutation (DDG) for protein-protein complexes.

## Installation

The model was tested on Linux using  `Python 3.8`, `PyTorch 2.1.2` ,`easydict 1.13` ,`numpy 1.26.4` ,`pandas 2.2.2`, and `Biopython 1.7.0`. The dependencies can be set up using the following commands:

```bash
git clone https://github.com/ShiweiWu-545/TGTPO.git
cd TGTPO

conda create --name TGTPO python=3.8
conda activate TGTPO
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Usage

The model requires two input PDB files: (1) a wild-type complex structure, and (2) a mutated complex structure. The mutated structures are typically built by protein design packages such as Rosetta. Note that both structures must have the same length. The DDG can be predicted for the two structures by running the command:

```bash
python ./run/run_model.py
```

## Citation

Coming soon...

## Contact

Please contact wushiwei@hrbeu.edu.cn for any questions related to the source code.
