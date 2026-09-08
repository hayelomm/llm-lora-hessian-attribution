"""
CAM-LLM preliminary artefact:
Hutchinson trace estimation for curvature-based parameter attribution.

The Hutchinson estimator approximates:

    Tr(H) ≈ (1/K) Σ z_k^T H z_k

where:
    H = Hessian of the loss
    z_k = random Rademacher vector

This implementation computes an approximate curvature score
for trainable parameters.
"""

import argparse

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def rademacher_like(tensor):
    """
    Generate a random Rademacher vector (+1/-1)
    with the same shape as tensor.
    """
    return torch.randint(
        0,
        2,
        tensor.shape,
        device=tensor.device,
        dtype=torch.int64,
    ).float() * 2 - 1


def compute_loss(model, tokenizer, text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    outputs = model(
        **inputs,
        labels=inputs["input_ids"],
    )

    return outputs.loss


def hutchinson_trace(
    model,
    tokenizer,
    text,
    samples=5,
):

    model.train()

    loss = compute_loss(
        model,
        tokenizer,
        text,
    )

    parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    first_grads = torch.autograd.grad(
        loss,
        parameters,
        create_graph=True,
        allow_unused=True,
    )

    trace_estimates = []

    for sample in range(samples):

        vectors = []

        for p in parameters:
            vectors.append(
                rademacher_like(p)
            )

        grad_dot_z = 0.0

        for grad, z in zip(first_grads, vectors):

            if grad is not None:
                grad_dot_z += (
                    grad * z
                ).sum()

        second_grads = torch.autograd.grad(
            grad_dot_z,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )

        z_h_z = 0.0

        for second_grad, z in zip(
            second_grads,
            vectors,
        ):

            if second_grad is not None:
                z_h_z += (
                    second_grad * z
                ).sum()

        trace_estimates.append(
            z_h_z.detach()
        )

    trace = torch.stack(
        trace_estimates
    ).mean()

    return trace.item()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="facebook/opt-1.3b",
    )

    parser.add_argument(
        "--text",
        default=(
            "Machine learning models can support "
            "intelligent decision making."
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),
    )

    model.to(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Computing Hutchinson curvature estimate...")

    score = hutchinson_trace(
        model,
        tokenizer,
        args.text,
        args.samples,
    )

    print(
        f"Approximate Hessian trace: {score:.6f}"
    )


if __name__ == "__main__":
    main()
