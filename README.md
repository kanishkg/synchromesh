# Synchromesh - Constrained Decoding from Language Models

This is an *unofficial* reimplementation of the Constrained Semantic Decoding (CSD) algorithm from the following paper: 

[arXiv link](https://arxiv.org/abs/2201.11227)

```bibtex
@inproceedings{poesia2022synchromesh,
  title={Synchromesh: Reliable Code Generation from Pre-trained Language Models},
  author={Poesia*, Gabriel and Polozov*, Alex and Le, Vu and Tiwari, Ashish and Soares, Gustavo and Meek, Christopher and Gulwani, Sumit},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

CSD allows you to sample from a language model (e.g., LLaMA 2, or OpenAI models that support the `Completions` API) while respecting constraints coming from a _Completion Engine_. A Completion Engine is a very flexible abstraction over a left-to-right constraint generator. One possible completion engine given here is a grammar engine derived from [Lark](https://github.com/lark-parser/lark). Using a completion engine derived from a Lark grammar, you can sample from a language model while guaranteeing that the output will be parseable.

More instructions will be here soon. As the API stabilized, we'll also upload the package to PyPI. For now, you can install it locally with:

```
$ python setup.py install
```
