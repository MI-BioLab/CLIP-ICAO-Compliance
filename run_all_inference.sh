#!/bin/bash
set -e

# The first argument is the path to the TONO dataset root
# The second argument is the mode (manual, pl-manual, pl-requirement)
# The third argument is the model (clip-iqa, oai-large14-336)
if [ $# -ne 3 ]; then
    echo "Usage: $0 <path_to_tono_dataset_root> <mode> <model>"
    exit 1
fi
TONO_ROOT="$1"
MODE="$2"
MODEL="$3"

tasks=("cap" "la" "ceg" "mkup" "sm" "sun" "tq" "expos" "oof" "sat" "bkg" "light" "pixel" "poster")

compliant=(
    "$TONO_ROOT/icao"
    "$TONO_ROOT/icao"
    "$TONO_ROOT/icao"
    "$TONO_ROOT/icao"
    "$TONO_ROOT/icao"
    "$TONO_ROOT/icao"
    "$TONO_ROOT/icao"
    "$TONO_ROOT/icao"
    "$TONO_ROOT/icao"
    "$TONO_ROOT/icao"
    "$TONO_ROOT/icao"
    "$TONO_ROOT/icao"
    "$TONO_ROOT/icao"
    "$TONO_ROOT/icao"
)
non_compliant=(
    "$TONO_ROOT/cap"
    "$TONO_ROOT/la_1 $TONO_ROOT/la_2"
    "$TONO_ROOT/ceg"
    "$TONO_ROOT/mkup"
    "$TONO_ROOT/sm"
    "$TONO_ROOT/sun"
    "$TONO_ROOT/tq"
    "$TONO_ROOT/expos"
    "$TONO_ROOT/oof"
    "$TONO_ROOT/sat"
    "$TONO_ROOT/bkg"
    "$TONO_ROOT/light"
    "$TONO_ROOT/pixel"
    "$TONO_ROOT/poster"
)

if [[ "$MODEL" == "clip-iqa" ]]; then
    model="clip-iqa"
elif [[ "$MODEL" == "oai-large14-336" ]]; then
    model="openai/clip-vit-large-patch14-336"
else
    echo "Unknown model: $MODEL. Use 'clip-iqa' or 'oai-large14-336'."
    exit 1
fi


if [[ "$MODE" == "manual" ]]; then
    # Manual prompts (both CLIP-IQA and oai-large14-336)
    positive_prompt=(
        "The subject is not wearing any type of headgear, and the hair is visible"
        "The subject is looking straight at the camera, the gaze is frontal"
        "The eyes are open and clearly visible, with the iris in view"
        "The subject is either not wearing makeup or has light, natural makeup that does not alter facial features"
        "The facial expression is natural, does not reveal any particular emotions, and does not alter the facial structure"
        "The subject is not wearing sun glasses that occlude the eyes making them not visible"
        "The head pose is frontal without rotations in terms of roll, pitch and yaw"
        "The photograph has proper exposure, neither too light nor too dark"
        "The photograph is in focus, and the fine details of the image are clearly visible"
        "The colors in the image are natural, neither too saturated nor too flat"
        "The background is uniform, with no irregularities, even small ones. It may have a slight color gradient in one direction."
        "The lighting on the face is even, with no shadows, not even small ones, including around the nose area. There are no shadows caused by glasses."
        "The image appears detailed and of good resolution, without visible pixelation or discretization effects"
        "The image is characterized by a good distribution of colors, with no visible discretization effects"
    )
    negative_prompt=(
        "The subject is wearing a hat, cap, bandana, or any other garment that hides the hair"
        "The subject is looking away, in a direction different from the camera"
        "The eyes are partially or fully closed, making them difficult to see clearly"
        "The subject is wearing heavy makeup, with bold, unnatural colors that alter facial features"
        "The subject's expression reflects an emotion and alters the geometry of the face, mouth, or eyes"
        "The subject wears sun glasses that hide the eyes"
        "The head is rotated, causing the facial features to appear asymmetric in at least one direction"
        "The exposure of the photograph is incorrect, resulting in an image that is either too light or too dark"
        "The photograph is blurred and smooth, no fine details are clearly visible"
        "The colors in the image are unnatural, either too saturated or too faded"
        "The background is not uniform and shows irregularities, streaks, or a complex background with various objects present"
        "The lighting is uneven, with shadows or areas of the face that are brighter than others"
        "The image is of low quality and highly quantized, with visible pixelation effects"
        "The image shows a posterization effect, with a noticeable discretization of colors"
    )
elif [[ "$MODEL" == "clip-iqa" && "$MODE" == "pl-manual" ]]; then
    # CLIP-IQA + manual PL prompts
    positive_prompt=(
        "Hair visible, no coverings"
        "Front view, eyes directly at camera"
        "Eyes clearly visible, open"
        "No visible makeup on eyes"
        "Unsmiling, neutral expression"
        "No sunglasses worn"
        "Symmetrical face"
        "The photo avoids exposure issues, keeping uniformity"
        "Picture has high clarity"
        "Photo colors untouched, natural"
        "Photo is genuine, not manipulated"
        "Face fully visible, unobstructed"
        "Sharp, well-defined image"
        "Image is sharp, no pixelation"
    )
    negative_prompt=(
        "Headwear hiding hair"
        "Side view, eyes elsewhere"
        "Eyes obstructed or closed"
        "Visible makeup on eyes"
        "Smiling or frowning"
        "Sunglasses hide eyes"
        "Asymmetric due to pose"
        "The photo has exposure issues, disrupting uniformity"
        "Picture has low clarity"
        "Photo colors unnatural, edited"
        "Photo is manipulated or modified"
        "Face partially visible, lens shadows"
        "Image is fuzzy, undefined"
        "Image is pixelated"
    )
elif [[ "$MODEL" == "clip-iqa" && "$MODE" == "pl-requirement" ]]; then
    # CLIP-IQA + requirement PL prompts
    positive_prompt=(
        "Eyes and brows visible"
        "Direct camera gaze, head upright"
        "Irises not shaded, eyes open"
        "Face without visible makeup"
        "Unsmiling, steady look"
        "No sunglasses during passport photo"
        "Front pose, eyes front"
        "Clear distinction between face and background"
        "Fine details sharp"
        "Naturally rendered colors"
        "Clear background, subject distinct"
        "Distinct forehead lines"
        "Portrait pixel density is consistent"
        "Photo captures necessary brightness contrast"
    )
    negative_prompt=(
        "Eyes and brows obscured by headwear"
        "Gaze away, head not straight"
        "Irises shaded, eyes closed"
        "Heavy makeup distorting features"
        "Big teeth showing, wide smile"
        "Wear sunglasses during passport photo"
        "Side pose, glance away"
        "Face blends with background"
        "Fine details blurred"
        "Altered rendered colors"
        "Subject blending into background"
        "Forehead lines lost in lighting"
        "Portrait pixel density is inconsistent"
        "Photo fails to capture necessary brightness contrast"
    )
elif [[ "$MODEL" == "oai-large14-336" && "$MODE" == "pl-manual" ]]; then
    # oai-large14-336 + manual PL prompts
    positive_prompt=(
        "Hair fully visible, no headwear"
        "Eyes centered, straight at camera"
        "Both eyes are open"
        "Bare face without obstructions"
        "Face shows no emotion, no structure change"
        "Eyes not hidden"
        "Face symmetry, no tilt or shift"
        "Photo is well-defined, clear image"
        "Photo is crisp and precise"
        "Image is unmodified"
        "Photo looks original, no digital effects"
        "Face not altered by light"
        "Photo is crisp, no noise"
        "Photo pristine, no effects"
    )
    negative_prompt=(
        "Head covered with a hat, cap, or bandana"
        "Eyes off-centered, looking aside"
        "One or both eyes closed"
        "Face with obstructive, vibrant makeup"
        "Facial features changed by smile"
        "Sunglasses obscure eyes"
        "Asymmetric from any rotation"
        "Photo is indistinct, lacking clarity"
        "Photo is blurry and imprecise"
        "Image modified or filtered"
        "Photo has digital effects"
        "Face lighting digitally altered"
        "Photo is fuzzy with noise"
        "Photo altered with digital effects"
    )
elif [[ "$MODEL" == "oai-large14-336" && "$MODE" == "pl-requirement" ]]; then
    # oai-large14-336 + requirement PL prompts
    positive_prompt=(
        "Face visible from ear to ear, no distortion"
        "Eyes locked onto the camera"
        "Pupils exposed, naturally open eyes"
        "Natural face"
        "Serious look"
        "Glasses with clear lenses"
        "Head aligned, camera"
        "Clear contrast in facial details"
        "Clear 1mm detail visibility"
        "Srgb standard capture"
        "Plain color, no designs"
        "Even brightness across facial features"
        "Portrait in high-resolution raw"
        "Photo meets intensity contrast needs, ear to ear"
    )
    negative_prompt=(
        "Ear to ear face covered by headwear"
        "Gazing away"
        "Pupils hidden, unnaturally closed eyes"
        "Heavy makeup"
        "Smiling"
        "Glasses with tinted lenses"
        "Head tilted away from camera"
        "Blurred contrast in facial details"
        "Blurry 1mm detail visibility"
        "Artificial colour capture"
        "Designed background"
        "Extreme brightness variation on face"
        "Portrait in low-resolution pdf"
        "Photo fails intensity contrast needs, ear to ear"
    )
else
    echo "Unknown mode: $MODE. Use 'manual', 'pl-manual', or 'pl-requirement'."
    exit 1
fi

for index in "${!tasks[@]}"; do
    echo "Running inference for task ${tasks[$index]}"
    python icao_single_prompt.py --model "$model" \
        --compliant-roots ${compliant[$index]} \
        --non-compliant-roots ${non_compliant[$index]} \
        --positive-prompts "${positive_prompt[$index]}" \
        --negative-prompts "${negative_prompt[$index]}"
done