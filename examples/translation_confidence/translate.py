"""
English -> Spanish translation test.

Reads every .txt file from an input subdirectory, asks the model to
translate its full contents into Spanish, and writes each translation
to a same-named file in an output subdirectory.

Usage:
    python translate.py [--model MODEL] [--input-dir DIR] [--output-dir DIR]
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = (
    "You are a professional English-to-Spanish translator. Translate the "
    "user's text into Spanish. Respond with only the translation — no "
    "notes, no explanations, no quotation marks."
)


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "translate.log"

    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def get_tags(model_name: str) -> tuple[str, str]:
    """User/assistant chat tags, keyed off model family (mirrors honesty.py)."""
    name = model_name.lower()
    if "wizard-vicuna" in name or "vicuna" in name:
        return "USER:", "ASSISTANT:"
    if "qwen" in name:
        return "<|im_start|>user", "<|im_end|>\n<|im_start|>assistant"
    return "[INST]", "[/INST]"


def build_prompt(text: str, user_tag: str, assistant_tag: str) -> str:
    if user_tag == "[INST]":
        return f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{text} [/INST]"
    return f"{user_tag} {SYSTEM_PROMPT}\n\n{text} {assistant_tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-7b-chat-hf",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--input-dir",
        default=str(SCRIPT_DIR / "input"),
        help="Directory containing English .txt files to translate",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "output"),
        help="Directory to write Spanish translations to",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=1024,
        help="Max tokens to generate per translation",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-translate files even if an output file already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    log = setup_logging(output_dir)

    if not input_dir.is_dir():
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    input_files = sorted(input_dir.glob("*.txt"))
    if not input_files:
        log.error("No .txt files found in %s", input_dir)
        sys.exit(1)

    log.info("=== Translation job starting ===")
    log.info("Model: %s", args.model)
    log.info("Input dir: %s (%d files)", input_dir, len(input_files))
    log.info("Output dir: %s", output_dir)

    log.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    use_fast = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=use_fast, padding_side="left", legacy=False
    )
    tokenizer.pad_token_id = 0
    log.info("Model loaded.")

    user_tag, assistant_tag = get_tags(args.model)
    log.info("Tags: user=%r  assistant=%r", user_tag, assistant_tag)

    for input_path in input_files:
        output_path = output_dir / input_path.name

        if output_path.exists() and not args.overwrite:
            log.info("Skipping %s (output already exists)", input_path.name)
            continue

        text = input_path.read_text(encoding="utf-8").strip()
        if not text:
            log.warning("Skipping %s (empty file)", input_path.name)
            continue

        prompt = build_prompt(text, user_tag, assistant_tag)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        log.info("Translating %s (%d chars)...", input_path.name, len(text))
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        translation = tokenizer.decode(generated, skip_special_tokens=True).strip()

        output_path.write_text(translation + "\n", encoding="utf-8")
        log.info("Wrote %s", output_path)

    log.info("=== Job complete ===")


if __name__ == "__main__":
    main()
