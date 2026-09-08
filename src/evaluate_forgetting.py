"""
Simple evaluation utility for sequential adaptation.

The script compares model loss/perplexity on a reference
dataset before and after sequential adaptation.

This provides a basic measurement of knowledge retention
and catastrophic forgetting.
"""

import argparse
import math

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


@torch.no_grad()
def evaluate(model, tokenizer, texts):

    model.eval()

    losses = []

    for text in texts:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }

        outputs = model(
            **inputs,
            labels=inputs["input_ids"],
        )

        losses.append(
            outputs.loss.item()
        )

    mean_loss = sum(losses) / len(losses)

    perplexity = math.exp(
        min(mean_loss, 20)
    )

    return mean_loss, perplexity


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_before",
        required=True,
    )

    parser.add_argument(
        "--model_after",
        required=True,
    )

    parser.add_argument(
        "--dataset",
        default="wikitext",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    dataset = load_dataset(
        args.dataset,
        "wikitext-2-raw-v1",
        split="test",
    )

    texts = [
        x["text"]
        for x in dataset
        if len(x["text"].strip()) > 20
    ][:args.samples]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_before
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Evaluating before adaptation...")

    model_before = (
        AutoModelForCausalLM
        .from_pretrained(args.model_before)
        .to(device)
    )

    loss_before, ppl_before = evaluate(
        model_before,
        tokenizer,
        texts,
    )

    del model_before

    print("Evaluating after adaptation...")

    model_after = (
        AutoModelForCausalLM
        .from_pretrained(args.model_after)
        .to(device)
    )

    loss_after, ppl_after = evaluate(
        model_after,
        tokenizer,
        texts,
    )

    print("\n========== RESULTS ==========")

    print(
        f"Before adaptation - "
        f"Loss: {loss_before:.4f}, "
        f"Perplexity: {ppl_before:.2f}"
    )

    print(
        f"After adaptation  - "
        f"Loss: {loss_after:.4f}, "
        f"Perplexity: {ppl_after:.2f}"
    )

    print(
        "\nChange in loss: "
        f"{loss_after - loss_before:.4f}"
    )


if __name__ == "__main__":
    main()
