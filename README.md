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

## Prompts

Here we report the original handcrafted prompt pairs, as well as the ones obtained via prompt learning from either the handcrafted prompts or the ISO/ICAO requirements, using CLIP-IQA or CLIP ViT-L/14@336 as the model.

### Handcrafted prompt pairs

| Requirement           | Positive prompt                                                                                                                                 | Negative prompt                                                                                                       |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| Head without covering | The subject is not wearing any type of headgear, and the hair is visible                                                                        | The subject is wearing a hat, cap, bandana, or any other garment that hides the hair                                  |
| Gaze in camera        | The subject is looking straight at the camera, the gaze is frontal                                                                              | The subject is looking away, in a direction different from the camera                                                 |
| Eyes open             | The eyes are open and clearly visible, with the iris in view                                                                                    | The eyes are partially or fully closed, making them difficult to see clearly                                          |
| No/light makeup       | The subject is either not wearing makeup or has light, natural makeup that does not alter facial features                                       | The subject is wearing heavy makeup, with bold, unnatural colors that alter facial features                           |
| Neutral expression    | The facial expression is natural, does not reveal any particular emotions, and does not alter the facial structure                              | The subject's expression reflects an emotion and alters the geometry of the face, mouth, or eyes                      |
| No sunglasses         | The subject is not wearing sun glasses that occlude the eyes making them not visible                                                            | The subject wears sun glasses that hide the eyes                                                                      |
| Frontal pose          | The head pose is frontal without rotations in terms of roll, pitch and yaw                                                                      | The head is rotated, causing the facial features to appear asymmetric in at least one direction                       |
| Correct exposure      | The photograph has proper exposure, neither too light nor too dark                                                                              | The exposure of the photograph is incorrect, resulting in an image that is either too light or too dark               |
| In focus photo        | The photograph is in focus, and the fine details of the image are clearly visible                                                               | The photograph is blurred and smooth, no fine details are clearly visible                                             |
| Correct saturation    | The colors in the image are natural, neither too saturated nor too flat                                                                         | The colors in the image are unnatural, either too saturated or too faded                                              |
| Uniform background    | The background is uniform, with no irregularities, even small ones. It may have a slight color gradient in one direction.                       | The background is not uniform and shows irregularities, streaks, or a complex background with various objects present |
| Uniform face lighting | The lighting on the face is even, with no shadows, not even small ones, including around the nose area. There are no shadows caused by glasses. | The lighting is uneven, with shadows or areas of the face that are brighter than others                               |
| No pixelation         | The image appears detailed and of good resolution, without visible pixelation or discretization effects                                         | The image is of low quality and highly quantized, with visible pixelation effects                                     |
| No posterization      | The image is characterized by a good distribution of colors, with no visible discretization effects                                             | The image shows a posterization effect, with a noticeable discretization of colors                                    |


### Learned prompt pairs for CLIP ViT-L/14@336 starting from handcrafted pairs

| Requirement           | Positive prompt                            | Negative prompt                          |
|-----------------------|--------------------------------------------|------------------------------------------|
| Head without covering | Hair fully visible, no headwear            | Head covered with a hat, cap, or bandana |
| Gaze in camera        | Eyes centered, straight at camera          | Eyes off-centered, looking aside         |
| Eyes open             | Both eyes are open                         | One or both eyes closed                  |
| No/light makeup       | Bare face without obstructions             | Face with obstructive, vibrant makeup    |
| Neutral expression    | Face shows no emotion, no structure change | Facial features changed by smile         |
| No sunglasses         | Eyes not hidden                            | Sunglasses obscure eyes                  |
| Frontal pose          | Face symmetry, no tilt or shift            | Asymmetric from any rotation             |
| Correct exposure      | Photo is well-defined, clear image         | Photo is indistinct, lacking clarity     |
| In focus photo        | Photo is crisp and precise                 | Photo is blurry and imprecise            |
| Correct saturation    | Image is unmodified                        | Image modified or filtered               |
| Uniform background    | Photo looks original, no digital effects   | Photo has digital effects                |
| Uniform face lighting | Face not altered by light                  | Face lighting digitally altered          |
| No pixelation         | Photo is crisp, no noise                   | Photo is fuzzy with noise                |
| No posterization      | Photo pristine, no effects                 | Photo altered with digital effects       |

### Learned prompt pairs for CLIP ViT-L/14@336 starting from ISO/ICAO requirement description

| Requirement           | Positive prompt                                  | Negative prompt                                  |
|-----------------------|--------------------------------------------------|--------------------------------------------------|
| Head without covering | Face visible from ear to ear, no distortion      | Ear to ear face covered by headwear              |
| Gaze in camera        | Eyes locked onto the camera                      | Gazing away                                      |
| Eyes open             | Pupils exposed, naturally open eyes              | Pupils hidden, unnaturally closed eyes           |
| No/light makeup       | Natural face                                     | Heavy makeup                                     |
| Neutral expression    | Serious look                                     | Smiling                                          |
| No sunglasses         | Glasses with clear lenses                        | Glasses with tinted lenses                       |
| Frontal pose          | Head aligned, camera                             | Head tilted away from camera                     |
| Correct exposure      | Clear contrast in facial details                 | Blurred contrast in facial details               |
| In focus photo        | Clear 1mm detail visibility                      | Blurry 1mm detail visibility                     |
| Correct saturation    | Srgb standard capture                            | Artificial colour capture                        |
| Uniform background    | Plain color, no designs                          | Designed background                              |
| Uniform face lighting | Even brightness across facial features           | Extreme brightness variation on face             |
| No pixelation         | Portrait in high-resolution raw                  | Portrait in low-resolution pdf                   |
| No posterization      | Photo meets intensity contrast needs, ear to ear | Photo fails intensity contrast needs, ear to ear |

### Learned prompt pairs for CLIP-IQA starting from handcrafted pairs

| Requirement           | Positive prompt                                      | Negative prompt                                      |
|-----------------------|------------------------------------------------------|------------------------------------------------------|
| Head without covering | Hair visible, no coverings                           | Headwear hiding hair                                 |
| Gaze in camera        | Front view, eyes directly at camera                  | Side view, eyes elsewhere                            |
| Eyes open             | Eyes clearly visible, open                           | Eyes obstructed or closed                            |
| No/light makeup       | No visible makeup on eyes                            | Visible makeup on eyes                               |
| Neutral expression    | Unsmiling, neutral expression                        | Smiling or frowning                                  |
| No sunglasses         | No sunglasses worn                                   | Sunglasses hide eyes                                 |
| Frontal pose          | Symmetrical face                                     | Asymmetric due to pose                               |
| Correct exposure      | The photo avoids exposure issues, keeping uniformity | The photo has exposure issues, disrupting uniformity |
| In focus photo        | Picture has high clarity                             | Picture has low clarity                              |
| Correct saturation    | Photo colors untouched, natural                      | Photo colors unnatural, edited                       |
| Uniform background    | Photo is genuine, not manipulated                    | Photo is manipulated or modified                     |
| Uniform face lighting | Face fully visible, unobstructed                     | Face partially visible, lens shadows                 |
| No pixelation         | Sharp, well-defined image                            | Image is fuzzy, undefined                            |
| No posterization      | Image is sharp, no pixelation                        | Image is pixelated                                   |

### Learned prompt pairs for CLIP-IQA starting from ISO/ICAO requirement description

| Requirement           | Positive prompt                               | Negative prompt                                      |
|-----------------------|-----------------------------------------------|------------------------------------------------------|
| Head without covering | Eyes and brows visible                        | Eyes and brows obscured by headwear                  |
| Gaze in camera        | Direct camera gaze, head upright              | Gaze away, head not straight                         |
| Eyes open             | Irises not shaded, eyes open                  | Irises shaded, eyes closed                           |
| No/light makeup       | Face without visible makeup                   | Heavy makeup distorting features                     |
| Neutral expression    | Unsmiling, steady look                        | Big teeth showing, wide smile                        |
| No sunglasses         | No sunglasses during passport photo           | Wear sunglasses during passport photo                |
| Frontal pose          | Front pose, eyes front                        | Side pose, glance away                               |
| Correct exposure      | Clear distinction between face and background | Face blends with background                          |
| In focus photo        | Fine details sharp                            | Fine details blurred                                 |
| Correct saturation    | Naturally rendered colors                     | Altered rendered colors                              |
| Uniform background    | Clear background, subject distinct            | Subject blending into background                     |
| Uniform face lighting | Distinct forehead lines                       | Forehead lines lost in lighting                      |
| No pixelation         | Portrait pixel density is consistent          | Portrait pixel density is inconsistent               |
| No posterization      | Photo captures necessary brightness contrast  | Photo fails to capture necessary brightness contrast |
