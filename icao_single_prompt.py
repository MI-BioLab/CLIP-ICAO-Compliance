import argparse
from pathlib import Path

from tqdm import tqdm
import torch
import torchvision.io as tio
import piq
from transformers import CLIPProcessor, CLIPModel

from utils import compute_tpr_fpr_accuracies


def load_image(filename: Path) -> torch.Tensor:
    img = tio.read_image(filename, mode="RGB").to(torch.float32)
    return img.unsqueeze(0) / 255.0


def get_model_and_processor(model_type: str) -> tuple[CLIPModel, CLIPProcessor]:
    if model_type == "clip-iqa":
        model = piq.clip_iqa.clip.load().eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    else:
        model = CLIPModel.from_pretrained(model_type).eval()
        processor = CLIPProcessor.from_pretrained(model_type)
    return model, processor


def encode_prompts(model: CLIPModel, processor: CLIPProcessor, prompt_pairs: list[tuple[str, str]], model_type: str, device: str | torch.device) -> torch.Tensor:
    with torch.inference_mode():
        prompts = [x for pair in prompt_pairs for x in pair]
        if model_type == "clip-iqa":
            text_processed = processor(text=prompts)
            anchors_text = torch.zeros(len(prompts), processor.tokenizer.model_max_length, dtype=torch.long, device=device)
            for i, tp in enumerate(text_processed["input_ids"]):
                anchors_text[i, :len(tp)] = torch.tensor(tp, dtype=torch.long, device=device)
            anchors = model.encode_text(anchors_text).float()
        else:
            text_processed = processor(text=prompts, return_tensors="pt", padding=True)
            anchors = model.get_text_features(text_processed["input_ids"].to(device), text_processed["attention_mask"].to(device)).float()
        anchors = anchors / anchors.norm(p=2, dim=-1, keepdim=True)
        anchors = anchors.view(len(prompt_pairs), 2, -1)
        return anchors


def encode_images(model: CLIPModel, processor: CLIPProcessor, images: torch.Tensor, model_type: str, device: str | torch.device) -> torch.Tensor:
    with torch.inference_mode():
        if model_type == "clip-iqa":
            default_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
            default_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
            images = (images - default_mean) / default_std
            img_features = model.encode_image(images.float(), pos_embedding=False).float()
        else:
            processed_images = processor(images=[i.cpu() for i in images], return_tensors="pt", padding=True, do_rescale=False)
            img_features = model.get_image_features(processed_images["pixel_values"].to(device)).float()
        return img_features / img_features.norm(p=2, dim=-1, keepdim=True)


def compute_scores(prompts_features: torch.Tensor, images_features: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        n_images = images_features.shape[0]
        prompt_pairs = prompts_features.shape[0]
        logits_per_image = 100 * images_features @ prompts_features.view(prompt_pairs * 2, -1).t()
        probs = logits_per_image.reshape(n_images, -1, 2).softmax(-1)[:, :, 0]
        return probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compliant-roots", type=Path, nargs="+")
    parser.add_argument("--non-compliant-roots", type=Path, nargs="+")
    parser.add_argument("--positive-prompt", type=str, required=True)
    parser.add_argument("--negative-prompt", type=str, required=True)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--model", type=str, default="clip-iqa")

    args = parser.parse_args()

    model, processor = get_model_and_processor(args.model)
    model = model.to("cuda:0")

    encoded_prompts = encode_prompts(model, processor, [(args.positive_prompt, args.negative_prompt)], device="cuda:0", model_type=args.model)

    compliant_files = []
    non_compliant_files = []

    if args.split_file is not None:
        valid_files = set()
        with open(args.split_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    valid_files.add(args.split_file.parent / line)
    else:
        valid_files = None
    for root in args.compliant_roots:
        if not root.exists():
            raise ValueError(f"Compliant root {root} does not exist.")
        found_files = [f for f in root.rglob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
        if valid_files is not None:
            found_files = [f for f in found_files if f in valid_files]
        if len(found_files) == 0:
            raise ValueError(f"No compliant files found in {root}.")
        compliant_files += found_files
    for root in args.non_compliant_roots:
        if not root.exists():
            raise ValueError(f"Non-compliant root {root} does not exist.")
        found_files = [f for f in root.rglob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
        if valid_files is not None:
            found_files = [f for f in found_files if f in valid_files]
        if len(found_files) == 0:
            raise ValueError(f"No non-compliant files found in {root}.")
        non_compliant_files += found_files

    y_pred = []
    y_true = []

    for filename in tqdm(compliant_files):
        img = load_image(filename)
        img = img.to("cuda:0")

        encoded_image = encode_images(model, processor, img, device="cuda:0", model_type=args.model)
        score = compute_scores(encoded_prompts, encoded_image)
        y_pred.append(score)
        y_true.append(1)

    for filename in tqdm(non_compliant_files):
        img = load_image(filename)
        img = img.to("cuda:0")

        encoded_image = encode_images(model, processor, img, device="cuda:0", model_type=args.model)
        score = compute_scores(encoded_prompts, encoded_image)
        y_pred.append(score)
        y_true.append(0)
    
    y_pred = torch.tensor(y_pred, device="cuda:0")
    y_true = torch.tensor(y_true, device="cuda:0")

    thresholds = torch.linspace(0, 1, 1000, device="cuda:0")
    tpr, fpr, acc, confusion = compute_tpr_fpr_accuracies(y_pred, thresholds, y_true, output_confusion=True)
    eer_idx = torch.argmin(torch.abs(fpr - (1 - tpr)))
    eer = (fpr[eer_idx] + (1 - tpr[eer_idx])) / 2
    print("EER:", eer.item())
    print("Threshold:", thresholds[eer_idx].item())
