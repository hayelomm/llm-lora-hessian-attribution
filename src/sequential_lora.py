"""
CAM-LLM preliminary artefact:
Sequential LoRA fine-tuning on a small causal language model.

This implementation demonstrates:
1. Loading a pretrained causal language model
2. Adding a LoRA adapter
3. Sequential adaptation on two tasks
4. Saving the adapted model after each stage

The implementation is intentionally lightweight and reproducible.
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


def load_text_dataset(dataset_name, split, text_column, max_samples):
    dataset = load_dataset(dataset_name, split=split)

    if max_samples is not None:
        dataset = dataset.select(
            range(min(max_samples, len(dataset)))
        )

    dataset = dataset.filter(
        lambda x: x[text_column] is not None
        and len(str(x[text_column]).strip()) > 0
    )

    return dataset


def tokenize_dataset(dataset, tokenizer, text_column, max_length):

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )


def train_stage(
    model,
    tokenizer,
    dataset,
    output_dir,
    stage_name,
    epochs,
    batch_size,
    learning_rate,
):

    stage_dir = os.path.join(output_dir, stage_name)

    training_args = TrainingArguments(
        output_dir=stage_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(stage_dir)
    tokenizer.save_pretrained(stage_dir)

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="facebook/opt-1.3b",
        help="Hugging Face model identifier",
    )

    parser.add_argument(
        "--dataset1",
        default="wikitext",
    )

    parser.add_argument(
        "--dataset2",
        default="wikitext",
    )

    parser.add_argument(
        "--output",
        default="checkpoints",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    print("Loading sequential training datasets...")

    dataset1 = load_text_dataset(
        args.dataset1,
        "train",
        "text",
        args.max_samples,
    )

    dataset2 = load_text_dataset(
        args.dataset2,
        "train",
        "text",
        args.max_samples,
    )

    dataset1 = tokenize_dataset(
        dataset1,
        tokenizer,
        "text",
        256,
    )

    dataset2 = tokenize_dataset(
        dataset2,
        tokenizer,
        "text",
        256,
    )

    print("\n========== STAGE 1 ==========")

    model = train_stage(
        model,
        tokenizer,
        dataset1,
        args.output,
        "stage1",
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )

    print("\n========== STAGE 2 ==========")

    model = train_stage(
        model,
        tokenizer,
        dataset2,
        args.output,
        "stage2",
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )

    print("\nSequential LoRA experiment completed.")

    print(
        f"Checkpoints saved to: {args.output}"
    )


if __name__ == "__main__":
    main()
