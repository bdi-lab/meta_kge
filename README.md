# Metagraph Learning

This code is an implementation of the paper, Knowledge Graph Embedding via Metagraph Learning.

This code is based on the [OpenKE](http://openke.thunlp.org) implementation, which is an open toolkit for knowledge graph embedding.

If you use this code, please cite our [paper](https://dl.acm.org/doi/abs/10.1145/3404835.3463072):
```
@inproceedings{chung-sigir2021,
  author = {Chung, Chanyoung and Whang, Joyce Jiyoung},
  title = {Knowledge Graph Embedding via Metagraph Learning},
  booktitle = {Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2021},
  pages = {2212--2216}
}
```

## Usage

### Building a Metagraph

To generate a metagraph with respect to the original graph, use `metagraph.py`.

```
python3 metagraph.py [data] [density]
```
`[data]`: name of the dataset (it should be same with the directory name of the dataset, contained in `./benchmarks` folder)
`[density]`: size of the metagraph

### Performing Metagraph Learning

To perform metagraph learning on certain dataset, use `meta_[model].py`.

For `TransE`, use

```
python3 meta_transe.py [data] [density] [alpha_meta] [margin_meta] [alpha] [margin]
```

For `DistMult`, use

```
python3 meta_distmult.py [data] [density] [alpha_meta] [regul_meta] [alpha] [regul]
```

For `RotatE`, use

```
python3 meta_rotate.py [data] [density] [alpha_meta] [margin_meta] [adv_meta] [alpha] [margin] [adv]
```