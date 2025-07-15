from pathlib import Path
import json
import argparse

from tqdm import tqdm
import torch
import torchvision.io as tio
import piq
from transformers import CLIPProcessor, CLIPModel
from llama_cpp import Llama
import pandas as pd

from utils import compute_eer, OrderedSet


MAX_LLM_ATTEMPTS = 100


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


def encode_prompts(model: CLIPModel, processor: CLIPProcessor, prompt_pairs: list[tuple[str, str]], model_type: str, device: str | torch.device) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    with torch.inference_mode():
        prompts = [x for pair in prompt_pairs for x in pair]
        if model_type == "clip-iqa":
            text_processed = processor(text=prompts)
            anchors_text = torch.zeros(len(prompts), processor.tokenizer.model_max_length, dtype=torch.long, device=device)
            n_tokens = []
            for i, tp in enumerate(text_processed["input_ids"]):
                anchors_text[i, :len(tp)] = torch.tensor(tp, dtype=torch.long, device=device)
                n_tokens.append(len(tp))
            anchors = model.encode_text(anchors_text).float()
        else:
            text_processed = processor(text=prompts, return_tensors="pt", padding=True)
            n_tokens = text_processed["attention_mask"].sum(dim=1).tolist()
            anchors = model.get_text_features(text_processed["input_ids"].to(device), text_processed["attention_mask"].to(device)).float()
        anchors = anchors / anchors.norm(p=2, dim=-1, keepdim=True)
        anchors = anchors.view(len(prompt_pairs), 2, -1)
        n_tokens = list(zip(n_tokens[::2], n_tokens[1::2]))
        return anchors, n_tokens


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


def sanitize_response(response: str) -> str:
    return response.strip().replace(".", "").lower()


def generate_initial_prompts_from_requirement(llm: Llama, requirement: str, n: int) -> list[tuple[str, str, str]]:
    result = OrderedSet()
    attempts = 0
    while len(result) < n and attempts < MAX_LLM_ATTEMPTS:
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content":
                        "You are a pattern learner. "
                        "You will read a requirement expressed in natural language, extracted from the official ISO/ICAO standards for compliance verification of passport photos. "
                        "You must generate exactly the specified number of pairs of positive and negative prompts for CLIP.\n"
                        "Rules:\n"
                        "- You must rephrase both positive and negative prompts.\n"
                        "- Do not change the topic of generated prompts compared to the requirement.\n"
                        "- Positive and negative prompts must be related to the same topic.\n"
                        "- Positive and negative prompts must be antonyms.\n"
                        "- Positive prompts must be compliant with the requirement, while negative prompts must be non-compliant.\n"
                        "- Do not repeat already presented prompt pairs.\n"
                        "- Do not repeat the prompt pair given as input.\n"
                        "- Each prompt must be shorter than 15 words.\n"
                        "- Do not exchange positive and negative prompts.\n"
                        "- Do not replicate verbatim the prompt pairs you have seen.\n"
                        "- Do not mention ISO/ICAO compliance explicitly in the prompts.\n"
                        "- Format output in JSON as in the following example: "
                        "[{'positive_prompt': '...', 'negative_prompt': '...'}, ...]"
                },
                {
                    "role": "user",
                    "content":
                        f"{requirement}\n\n"
                        f"Generate {n} related prompts following the rules."
                }
            ],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "positive_prompt": { "type": "string" },
                            "negative_prompt": { "type": "string" }
                        },
                        "required": ["positive_prompt", "negative_prompt"]
                    }
                }
            },
            temperature=1.0
        )
        try:
            generated_prompts = json.loads(response["choices"][0]["message"]["content"])
            generated_prompts = [(sanitize_response(x["positive_prompt"]), sanitize_response(x["negative_prompt"]), "exploration") for x in generated_prompts]
        except (json.JSONDecodeError, KeyError):
            generated_prompts = []
        result.update(generated_prompts)
        print(f"Attempt {attempts + 1}/{MAX_LLM_ATTEMPTS}, generated {len(result)}/{n}")
        attempts += 1
    if len(result) < n:
        print(f"Failed to generate enough prompts after {MAX_LLM_ATTEMPTS} attempts, returning what was generated.")
    return list(result)[:n]


def generate_initial_prompts(llm: Llama, seed_prompt: tuple[str, str], *, n: int) -> list[tuple[str, str, str]]:
    seed_positive, seed_negative = seed_prompt
    result = OrderedSet()
    attempts = 0
    seed_json = json.dumps({"positive_prompt": seed_positive, "negative_prompt": seed_negative})
    while len(result) < n and attempts < MAX_LLM_ATTEMPTS:
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content":
                        "You are a pattern learner. "
                        "You will read a pair of antonym prompts expressed in natural language, extracted from the official ISO/ICAO standards for compliance verification of passport photos. "
                        "You must generate exactly the specified number of pairs of positive and negative prompts for CLIP.\n"
                        "Rules:\n"
                        "- You must rephrase both positive and negative prompts.\n"
                        "- Do not change the topic of generated prompts compared to the used-provided prompts.\n"
                        "- Positive and negative prompts must be related to the same topic.\n"
                        "- Positive and negative prompts must be antonyms.\n"
                        "- Positive prompts must be compliant with the requirement, while negative prompts must be non-compliant.\n"
                        "- Do not repeat already presented prompt pairs.\n"
                        "- Do not repeat the prompt pair given as input.\n"
                        "- Each prompt must be shorter than 15 words.\n"
                        "- Do not exchange positive and negative prompts.\n"
                        "- Do not replicate verbatim the prompt pairs you have seen.\n"
                        "- Format output in JSON as in the following example: "
                        "[{'positive_prompt': '...', 'negative_prompt': '...'}, ...]"
                },
                {
                    "role": "user",
                    "content":
                        f"{seed_json}\n\n"
                        f"Generate {n} related prompts following the rules."
                }
            ],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "positive_prompt": { "type": "string" },
                            "negative_prompt": { "type": "string" }
                        },
                        "required": ["positive_prompt", "negative_prompt"]
                    }
                }
            },
            temperature=1.0
        )
        try:
            generated_prompts = json.loads(response["choices"][0]["message"]["content"])
            generated_prompts = [(sanitize_response(x["positive_prompt"]), sanitize_response(x["negative_prompt"]), "exploration") for x in generated_prompts]
        except (json.JSONDecodeError, KeyError):
            generated_prompts = []
        result.update(generated_prompts)
        print(f"Attempt {attempts + 1}/{MAX_LLM_ATTEMPTS}, generated {len(result)}/{n}")
        attempts += 1
    if len(result) < n:
        print(f"Failed to generate enough prompts after {MAX_LLM_ATTEMPTS} attempts, returning what was generated.")
    return list(result)[:n]


def generate_prompts_offspring(
    llm: Llama,
    best_k: list[tuple[tuple[str, str, str], float]],
    worst_k: list[tuple[tuple[str, str, str], float]] | None,
    *,
    n: int,
    seed_prompt: tuple[str, str] | None = None,
    requirement: str | None = None,
    exploration_ratio: float,
    include_best_k: bool,
) -> list[tuple[str, str, str]]:
    def _generate_template(best: bool, prompts: list[tuple[tuple[str, str, str], float]]) -> str:
        adjective = "good" if best else "bad"
        result = f"List of {len(prompts)} {adjective.lower()} prompt pairs, ranked from lowest to highest EER:\n"
        for idx, ((positive, negative, _), eer) in enumerate(prompts):
            result += f"{idx + 1}. {{\"positive_prompt\": \"{positive}\", \"negative_prompt\": \"{negative}\"}} = EER: {eer:.8f}\n"
        return result

    result = OrderedSet(x[0] for x in best_k) if include_best_k else OrderedSet()
    n_exploration_prompts = int(n * exploration_ratio)

    # Generate exploration prompts starting either from the initial seed prompt or the requirement
    if n_exploration_prompts > 0:
        print("Generating new exploration prompts...")
        if seed_prompt is not None:
            seed_positive, seed_negative = seed_prompt
            exploration_prompts = generate_initial_prompts(llm, seed_prompt, n=n_exploration_prompts)
            seed_json = json.dumps({"positive_prompt": seed_positive, "negative_prompt": seed_negative})
            remember = f"Remember the initial seed prompts:\n{seed_json}\n\n"
        elif requirement is not None:
            exploration_prompts = generate_initial_prompts_from_requirement(llm, requirement, n=n_exploration_prompts)
            remember = f"Remember the initial requirement:\n{requirement}\n\n"
        else:
            raise ValueError("Invalid configuration: either seed prompts or requirement must be provided, but not both.")
        result.update(exploration_prompts)
    else:
        remember = ""

    print("Generating offspring prompts...")
    attempts = 0
    if worst_k is not None:
        generation_rule = "Write better prompts inspired by the good ones, while avoiding the bad ones."
        template = f"{_generate_template(True, best_k)}\n\n{_generate_template(False, worst_k)}\n\n"
    else:
        generation_rule = "Write better prompts inspired by the good ones."
        template = f"{_generate_template(True, best_k)}\n\n"
    while len(result) < n and attempts < MAX_LLM_ATTEMPTS:
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content":
                        "You are a pattern learner. "
                        "You must generate exactly the specified number of pairs of positive and negative prompts for CLIP.\n"
                        "Rules:\n"
                        "- You will see a certain number of good prompt pairs and bad prompt pairs.\n"
                        "- Based on these patterns, your objective is to minimize the Equal Error Rate (EER) associated with the prompts.\n"
                        "- Lower Equal Error Rate (EER) is better.\n"
                        f"- {generation_rule}\n"
                        "- You must rephrase both positive and negative prompts.\n"
                        "- Do not change the topic of generated prompts compared to the user-provided prompts.\n"
                        "- Positive and negative prompts must be related to the same topic.\n"
                        "- Positive and negative prompts must be antonyms.\n"
                        "- Do not repeat already presented prompt pairs.\n"
                        "- Do not repeat the prompt pair given as input.\n"
                        "- Each prompt must be shorter than 15 words.\n"
                        "- Do not exchange positive and negative prompts.\n"
                        "- Do not replicate verbatim the prompt pairs you have seen.\n"
                        "- Format output in JSON as in the following example: "
                        "[{'positive_prompt': '...', 'negative_prompt': '...'}, ...]"
                },
                {
                    "role": "user",
                    "content":
                        f"{template}"
                        f"Generate {n} similar prompts following the rules.\n"
                        f"{remember}"
                }
            ],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "positive_prompt": { "type": "string" },
                            "negative_prompt": { "type": "string" }
                        },
                        "required": ["positive_prompt", "negative_prompt"]
                    }
                }
            },
            temperature=1.0
        )
        try:
            generated_prompts = json.loads(response["choices"][0]["message"]["content"])
            generated_prompts = [(sanitize_response(x["positive_prompt"]), sanitize_response(x["negative_prompt"]), "exploitation") for x in generated_prompts]
        except (json.JSONDecodeError, KeyError):
            generated_prompts = []
        result.update(generated_prompts)
        print(f"Attempt {attempts + 1}/{MAX_LLM_ATTEMPTS}, generated {len(result)}/{n}")
        attempts += 1
    if len(result) < n:
        print(f"Failed to generate enough prompts after {MAX_LLM_ATTEMPTS} attempts, returning what was generated.")
    return list(result)[:n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirement-file", type=Path, help="File containing the requirement to generate prompts for.")
    parser.add_argument("--positive-seed-prompt", type=str, help="Positive seed prompt to generate prompts from.")
    parser.add_argument("--negative-seed-prompt", type=str, help="Negative seed prompt to generate prompts from.")
    parser.add_argument("--seed-prompts-file", type=Path, help="File containing the seed prompts to generate prompts from.")
    parser.add_argument("--compliant-roots", type=Path, nargs="+", help="Root directories containing compliant images.")
    parser.add_argument("--non-compliant-roots", type=Path, nargs="+", help="Root directories containing non-compliant images.")
    parser.add_argument("--split-file", type=Path, help="File containing the split of compliant and non-compliant images that should be used.")
    parser.add_argument("--n-generations", type=int, default=20, help="Number of generations to run the optimization for.")
    parser.add_argument("--n-prompts", type=int, default=200, help="Number of prompts to generate in each generation.")
    parser.add_argument("--keep-top-k-prompts", type=int, default=10, help="Number of top prompts to keep.")
    parser.add_argument("--exploitation-k-prompts", type=int, default=10, help="Number of top/bottom prompts to use for exploitation.")
    parser.add_argument("--exploration-ratio", type=float, default=0.5, help="Ratio of exploration prompts to generate.")
    parser.add_argument("--exploration-top-only", action="store_true", help="Only use top prompts for exploration instead of both top and bottom prompts.")
    parser.add_argument("--spare-top-k", action="store_true", help="Spare the top k prompts from being replaced.")
    parser.add_argument("--llm-model-path", type=Path, required=True, help="Path to the Llama.cpp-compatible model.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on.")
    parser.add_argument("--export-history", type=Path, help="File to export the prompts history to in CSV format.")
    parser.add_argument("--model", type=str, default="clip-iqa", help="Model to use for encoding images and prompts.")

    args = parser.parse_args()

    device = torch.device(args.device)
    use_seed_prompts = args.requirement_file is None
    if use_seed_prompts:
        requirement = None
        if args.seed_prompts_file is not None:
            seed_prompts = args.seed_prompts_file.read_text().strip().split("\n")
            if len(seed_prompts) != 2:
                raise ValueError("Seed prompts file must contain exactly two prompts.")
            seed_prompts = tuple(seed_prompts)
        elif args.positive_seed_prompt is None and args.negative_seed_prompt is None:
            raise ValueError("Either seed prompts file or seed prompts must be provided.")
        else:
            seed_prompts = (args.positive_seed_prompt, args.negative_seed_prompt)
    else:
        seed_prompts = None
        requirement = args.requirement_file.read_text().strip().replace("\n", " ")
    n_generations = args.n_generations
    n_prompts = args.n_prompts
    keep_top_k_prompts = args.keep_top_k_prompts
    exploitation_k_prompts = args.exploitation_k_prompts
    exploration_ratio = args.exploration_ratio
    exploration_top_only = args.exploration_top_only
    spare_top_k = args.spare_top_k
    model_type = args.model

    files, gt = [], []

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
        files += found_files
        gt += [1] * len(found_files)
    for root in args.non_compliant_roots:
        if not root.exists():
            raise ValueError(f"Non-compliant root {root} does not exist.")
        found_files = [f for f in root.rglob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
        if valid_files is not None:
            found_files = [f for f in found_files if f in valid_files]
        if len(found_files) == 0:
            raise ValueError(f"No non-compliant files found in {root}.")
        files += found_files
        gt += [0] * len(found_files)
    assert len(files) == len(gt)

    model, processor = get_model_and_processor(model_type)
    model = model.to(device)

    image_encodings = []
    for filename in tqdm(files):
        img = load_image(filename).to(device)
        encoded = encode_images(model, processor, img, device=device, model_type=model_type)
        image_encodings.append(encoded)
    image_encodings = torch.cat(image_encodings, dim=0).to(device)
    y_true = torch.tensor(gt).to(device)

    llm = Llama(model_path=str(args.llm_model_path), n_gpu_layers=-1, n_ctx=16384, use_mmap=False, verbose=False)
    print("Generating initial prompts...")
    if use_seed_prompts:
        initial_prompts = generate_initial_prompts(llm, seed_prompt=seed_prompts, n=n_prompts)
    else:
        initial_prompts = generate_initial_prompts_from_requirement(llm, requirement=requirement, n=n_prompts)

    prompts_history = []
    surviving_prompts = initial_prompts

    for generation in range(n_generations):
        to_encode = [(positive, negative) for positive, negative, _ in surviving_prompts]
        prompts_encodings, prompts_n_tokens = encode_prompts(model, processor, prompt_pairs=to_encode, device=device, model_type=model_type)
        y_pred = compute_scores(prompts_encodings, image_encodings)

        # Save the EER for each prompt pair
        prompts_eer = []
        for prompt_pair_idx in range(len(surviving_prompts)):
            eer, _ = compute_eer(y_pred[:, prompt_pair_idx], y_true)
            prompts_eer.append(eer)

        # Keep track of best and worst prompts for exploitation phase
        sorted_prompts = sorted(zip(surviving_prompts, prompts_eer), key=lambda x: x[1])
        best_k = sorted_prompts[:exploitation_k_prompts]
        worst_k = sorted_prompts[-exploitation_k_prompts:]
        print(f"Generation {generation + 1}:")
        for i, ((positive, negative, phase), eer) in enumerate(best_k):
            print(f"    Best {i + 1}: (\"{positive}\", \"{negative}\") EER: {eer:.4f}, Phase: {phase}")
        for i, ((positive, negative, phase), eer) in enumerate(worst_k):
            print(f"    Worst {i + 1}: (\"{positive}\", \"{negative}\") EER: {eer:.4f}, Phase: {phase}")
        
        # Save all generated prompts, their EER, and the number of tokens for each prompt
        for prompt_pair, eer, n_tokens in zip(surviving_prompts, prompts_eer, prompts_n_tokens):
            prompts_history.append((prompt_pair, generation, eer, n_tokens))

        # Avoid generating new prompts if this is the last generation
        if generation == n_generations - 1:
            break

        # Generate the offspring
        if use_seed_prompts:
            surviving_prompts = generate_prompts_offspring(
                llm,
                best_k,
                worst_k if not exploration_top_only else None,
                n=n_prompts,
                seed_prompt=seed_prompts,
                exploration_ratio=exploration_ratio,
                include_best_k=spare_top_k,
            )
        else:
            surviving_prompts = generate_prompts_offspring(
                llm,
                best_k,
                worst_k if not exploration_top_only else None,
                n=n_prompts,
                requirement=requirement,
                exploration_ratio=exploration_ratio,
                include_best_k=spare_top_k,
            )

    print(f"Best {keep_top_k_prompts} prompts found:")
    # Remove duplicate prompts across generations
    seen_prompts = set()
    unique_prompts = []
    for prompt_pair, generation, eer, n_tokens in prompts_history:
        positive, negative, _ = prompt_pair
        if (positive, negative) not in seen_prompts:
            seen_prompts.add((positive, negative))
            unique_prompts.append((prompt_pair, eer, n_tokens, generation))
    unique_prompts = sorted(unique_prompts, key=lambda x: x[1])
    for i, ((positive, negative, phase), eer, n_tokens, generation) in enumerate(unique_prompts[:keep_top_k_prompts]):
        print(f"    {i + 1}: (\"{positive}\", \"{negative}\") EER: {eer:.4f} Tokens: {n_tokens} Generation: {generation + 1}, Phase: {phase}")

    if args.export_history is not None:
        df = pd.DataFrame(
            {
                "positive": p,
                "negative": n,
                "generation": g,
                "phase": ph,
                "eer": e.item(),
                "positive_n_tokens": pt,
                "negative_n_tokens": pn
            }
            for (p, n, ph), g, e, (pt, pn) in prompts_history
        )
        df.to_csv(args.export_history, index=False)

    print("Done.")
