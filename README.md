# Towards Fully-Automated and Zero-Shot ISO/ICAO Compliance Verification via CLIP-IQA and Natural Language Processing

This repository contains the code and models for our IJCB 2025 paper "Towards Fully-Automated and Zero-Shot ISO/ICAO Compliance Verification via CLIP-IQA and Natural Language Processing".

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MI-BioLab/CLIP-ICAO-Compliance.git
   cd CLIP-ICAO-Compliance
   ```
2. Create the Anaconda environment:
   ```bash
   conda env create -f environment.yml
   conda activate CLIP-ICAO-Compliance
   ```
3. Download and unpack the TONO dataset with the splits, which can be found [here](https://miatbiolab.csr.unibo.it/wp-content/uploads/downloads/TONO_split.zip).

## Usage

To run the experiments for ISO/ICAO face compliance verification, use the provided Bash script `run_all_inference.sh`:

```bash
./run_all_inference.sh <path_to_tono_dataset> <mode> <model>
```

The script requires three arguments:
* The first argument is the path to the TONO dataset root directory.
* The second argument is the mode, which can be `manual` (to use the handcrafted prompt pairs), `pl-manual` (to use the prompt pairs obtained via prompt learning starting from the handcrafted prompt pairs), or `pl-requirement` (to use the prompt pairs obtained via prompt learning starting from the official ISO/ICAO requirements).
* The third argument is the model, which can be `clip-iqa` or `oai-large14-336`.

If you want to run the Python verficiation script directly, you can use the following command:

```bash
python icao_single_prompt.py --compliant-roots <list of directories containing compliant images> --non-compliant-roots <list of directories containing non-compliant images> --positive-prompt <positive prompt> --negative-prompt <negative prompt> --split-file <path to split file> --model <model>
```

## Prompt learning

To perform prompt learning, you can use the provided Bash script `run_all_prompt_learning_tono.sh`:

```bash
./run_all_prompt_learning_tono.sh <path_to_tono_dataset>
```

This script requires one argument, which is the path to the TONO dataset root directory.
This will run both models (`clip-iqa` and `oai-large14-336`) and both seeding techniques (from handcrafted prompt pairs and from official ISO/ICAO requirements).

If you want to run the prompt learning algorithm directly, use the `prompt_learning.py` script.
Before starting it, ensure that you have downloaded and moved the LLM weights in the correct directory.
See details [here](models/README.md).

The script requires the following arguments:
* `--model`: The model to use for CLIP/CLIP-IQA, either `clip-iqa` or expressed as HuggingFace model name, e.g., `openai/clip-vit-large-patch14-336`
* `--compliant-roots`: A list of directories containing compliant images.
* `--non-compliant-roots`: A list of directories containing non-compliant images.
* `--split-file`: The path to the split file.
* `--n-generations`: The number of generations to perform prompt learning.
* `--n-prompts`: The number of prompts to generate at each generation.
* `--keep-top-k-prompts`: The number of top prompts to keep at the end of the prompt learning process.
* `--exploitation-k-prompts`: The number of top/bottom prompts to use for exploitation.
* `--exploration-ratio`: The ratio of exploration prompts to use compared against the total number of prompts.
* `--exploration-top-only`: If set, only the top prompts will be used for exploration.
* `--spare-top-k`: If set, the top k prompts will be kept for the next generation.
* `--llm-model-path`: The path to the LLM model weights, in GGUF format.
* `--device`: The device to use for running the LLM model and CLIP/CLIP-IQA, e.g., `cuda:0`.
* `--export-history`: The file to export the prompts history to in CSV format.

Furthermore, you can provide the seed prompt pairs either as string or as a file, with the first line being the positive prompt and the second line being the negative prompt:
* `--positive-seed-prompt`: The positive seed prompt to use.
* `--negative-seed-prompt`: The negative seed prompt to use.
* `--seed-prompts-file`: The path to the file containing the seed prompts.

If you want to start the prompt learning from the official ISO/ICAO requirements, you can use the `--requirement-file` argument:
* `--requirement-file`: The official ISO/ICAO requirement to use as a seed for prompt learning.

## Citation

If you use this code or the models, please cite the following paper:

```bibtex
@inproceedings{di2025towards,
   title={Towards Fully-Automated and Zero-Shot ISO/ICAO Compliance Verification via CLIP-IQA and Natural Language Processing},
   author={Di Domenico, Nicol{\`o} and Borghi, Guido and Franco, Annalisa and Maltoni, Davide},
   booktitle={2025 IEEE International Joint Conference on Biometrics (IJCB)},
   pages={1--10},
   year={2025},
   organization={IEEE},
}
```
