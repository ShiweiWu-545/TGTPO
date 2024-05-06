# TGTPO

![image](https://github.com/ShiweiWu-545/TGTPO/blob/main/data/overview.jpg)

TGTPO a deep-learning framework designed to uncover the functional impacts of mutations on the affinities of PPIs. This framework differs from other transformer-based models typically applied to tasks such as image recognition and protein structure prediction29-34. Our framework consists of two main components: a dynamic residue selection strategy that captures information on complete affinity changes as input; and a three-pass neural network module called MIDFORMER, which serves as the backbone of TGTPO and is used to model the dependencies and information interactions between multi-scale and multi-level features.

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
