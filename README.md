# Detecting Notational Errors in Digital Music Scores

[![HAL Badge](https://img.shields.io/badge/hal-05294807-00005c)](https://hal.science/hal-05294807)
[![ArXiv Badge](https://img.shields.io/badge/arXiv-2510.02746-b31b1b)](https://www.arxiv.org/abs/2510.02746)

This repository contains the official code to reproduce the results of the article
["Detecting Notational Errors in Digital Music Scores", TENOR 2025, L. Géré, N. Audebert, and F. Jacquemard](https://hal.science/hal-05294807).

## Installation

The code has been tested with Python 3.11, and we recommend running it in a virtual environment.
You can install the package and its dependencies using pip:
```bash
git clone https://github.com/leleogere/detecting-notational-errors-in-digital-music-scores
cd detecting-notational-errors-in-digital-music-scores
pip install .
```

## Usage

You can then use the CLI tool `detect_errors` to check one or more MusicXML files for notational errors.
There are two subcommands available:
- `check`: Check MusicXML files for errors, and output the results in a JSON file.
- `stats`: Display some statistics about an error file produced by the `check` command.

You can see the options for each command with the `--help` flag.

Here is how to check a single MusicXML file:
```bash
detect_errors check --mode both --path score.musicxml --output output.json
```
The mode `both` will run sequentially the individual and contextual checkers.
The option `--path` can also be a directory containing multiple MusicXML files, or a text file listing the paths of MusicXML files to check.

## Dataset and reproduction

The dataset used in this work is the [ASAP dataset](https://github.com/fosfrancesco/asap-dataset).
Note that to reproduce the results of the paper, you will need the same version of the dataset as used in our experiments
(commit [afc815c75c42e83a79c03feb6da8a35e77d4c6b8](https://github.com/fosfrancesco/asap-dataset/tree/afc815c75c42e83a79c03feb6da8a35e77d4c6b8)).
You can download it as follows:
```bash
git clone https://github.com/fosfrancesco/asap-dataset
cd asap-dataset
git checkout afc815c75c42e83a79c03feb6da8a35e77d4c6b8
```

You will also need the exact version of Partitura specified in the `pyproject.toml` file (1.7.0), 
as some errors can be caused by Partitura issues.

To reproduce the results in table 2 of the paper, simply run the following command:
```bash
detect_errors check --mode both --path path/to/asap-dataset --output all_errors.json --statistics
```

## Fixed scores

Some scores were fixed manually during the experiments.
They have been committed to the ASAP dataset repository, for now only in the
[`develop` branch](https://github.com/fosfrancesco/asap-dataset/tree/develop).

You can find the relevant pull requests here:
[#12](https://github.com/fosfrancesco/asap-dataset/pull/12),
[#13](https://github.com/fosfrancesco/asap-dataset/pull/13),
[#14](https://github.com/fosfrancesco/asap-dataset/pull/14),
[#15](https://github.com/fosfrancesco/asap-dataset/pull/15),
[#16](https://github.com/fosfrancesco/asap-dataset/pull/16),
[#17](https://github.com/fosfrancesco/asap-dataset/pull/17),
[#19](https://github.com/fosfrancesco/asap-dataset/pull/19),
[#20](https://github.com/fosfrancesco/asap-dataset/pull/20),
[#22](https://github.com/fosfrancesco/asap-dataset/pull/22),
[#23](https://github.com/fosfrancesco/asap-dataset/pull/23),
[#24](https://github.com/fosfrancesco/asap-dataset/pull/24).

## Cite

If you use this work in your research, please cite the following paper:
```bibtex
@inproceedings{gere_tenor2025,
  title = {{Detecting Notational Errors in Digital Music Scores}},
  author = {Géré, Léo and Audebert, Nicolas and Jacquemard, Florent},
  url = {https://hal.science/hal-05294807},
  booktitle = {{International Conference on Technologies for Music Notation and Representation (TENOR) 2025}},
  address = {Beijing, China},
  year = {2025},
  month = Oct,
  pdf = {https://hal.science/hal-05294807v1/file/paper.pdf},
}
```