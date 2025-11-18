# NLP-Based Semantic Matching on ECLASS

## About this Project 

This repository contains the implementation and evaluation code for my master's thesis _"NLP-Based Semantic Matching on ECLASS: Design and Validation of an Industrie 4.0 Matching Service"_ at the Chair of Information and Automation Systems for Process and Material Technology at RWTH Aachen University.

The aim is to develop a proof-of-concept Semantic Matching Service that leverages NLP techniques to semantically match concept definitions from the [IEC 61360-2](https://webstore.iec.ch/en/publication/5381)-compliant [ECLASS](https://eclass.eu/en/eclass-standard/introduction) dictionary and to make the matchings accessible via a [Semantic Matching Registry](https://github.com/s-heppner/semantic-alignment-core). Furthermore, the project aims to investigate occurring matching patterns, outliers and errors.

## Project Structure

```
semantic-matching-nlp-eclass/
│
├── data/                           # All data
│   ├── embedded-classes/           # Embeddings of all classes
│   ├── extracted-classes/          # Extracted classes from original files
│   ├── extracted-properties/       # Extracted properties from original files
│   ├── original/                   # Original data
│   └── scores/                     # Matching scores
│
├── logs/                           # Log files of important experiments
│
├── src/                            # Source code
│   ├── logic/                      # Embeddings generation and fuzzy logic core
│   ├── preprocessing/              # Data preprocessing
│   ├── registry/                   # Connection to the Semantic Matching Registry
│   └── utils/                      # Helper functions
│
├── test/                           # Unit testing
│
├── test_data/                      # Data for testing
│
└── visualisation/                  # Visualised results
```

Please note that, due to ECLASS copyright restrictions, files in `data/` and `visualisation/` cannot be included in this public repository.

## Getting Started
How to get this project running, assuming you have a cuda-capable GPU on your Windows machine:
- Run `nvidia-smi` to see which version of cuda you have (For me that was cuda v12.9) 
- Visit https://pytorch.org/get-started/locally/ and select the correct properties to get an installation link that looks like this:

```commandline
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
- To verify if it worked, try the following snippet:

```python
import torch
torch.cuda.is_available()
```

- Run `pip install -r requirements.txt` (Note that pip was a bit unhappy with the pytorch dependency, but since we already installed it above I quickly commented it out.)

- Add the raw ECLASS Basic files (`ECLASS15_0_BASIC_EN_SG_01.xml`) to the `./data/raw/` directory
- Run `src/data/extract_xml_to_csv.py`
- Run `src/data/embeddings_<model>.py`

>[!note]
> Theoretically, the script is supposed to download the necessary model files, however I had to manually download the models 
> and use a local filepath, since the script download kept getting stuck for some reason.
> If that's the case, you can simply edit the line: `model = SentenceTransformer("<put path here instead of model name>, ...)`.
