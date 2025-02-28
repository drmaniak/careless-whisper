#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
import wandb
import yaml
from datasets import Audio, Features, Value, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    GenerationConfig,
    Seq2SeqTrainer,
    TrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


# Define a custom data collator that converts raw samples into a batch.
def data_collator(features):
    """
    Converts a list of samples into a batch.
    Each sample should have:
      - "audio_file": file path for the audio (lazy-loaded by the Audio feature)
      - "text": the target transcript with <|speakerturn|> tokens.
    """
    input_features_list = []
    labels_list = []
    for f in features:
        # Use the processor to convert audio to model inputs.
        inputs = processor(
            f["audio_file"]["array"],
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
        )
        input_features_list.append(inputs.input_features.squeeze(0))
        # Tokenize the text target.
        tokenized = processor.tokenizer(
            f["text"],
            truncation=True,
            max_length=448,
            padding="max_length",
            return_tensors="pt",
        )
        labels_list.append(tokenized.input_ids.squeeze(0))
    batch = {
        "input_features": torch.stack(input_features_list),
        "labels": torch.stack(labels_list),
    }
    return batch


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(config_path):
    # Load configuration from YAML.
    config = load_config(config_path)

    # Initialize wandb (Weights & Biases) for logging.
    wandb.init(
        project=config.get("wandb_project", "whisper-diarization"), config=config
    )

    # Define project directories.
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / "data"
    OUTPUT_DIR = ROOT_DIR / "outputs"
    CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

    # Load training and validation datasets.
    train_ds = load_from_disk(DATA_DIR / "ami_dataset_train")
    val_ds = load_from_disk(DATA_DIR / "ami_dataset_validation")

    # Load the pre-trained Whisper model and processor.
    model_name = config.get("model_name", "openai/whisper-small")
    global processor  # Make processor available for data_collator.
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Add the special diarization token.
    special_tokens = {"additional_special_tokens": ["<|speakerturn|>"]}
    processor.tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(processor.tokenizer))
    print("Special tokens:", processor.tokenizer.additional_special_tokens)

    # Configure LoRA adapters.
    lora_config = LoraConfig(
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        target_modules=["q_proj", "v_proj"],
        lora_dropout=config.get("lora_dropout", 0.1),
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    print("LoRA adapters applied.")

    # Set up TrainingArguments with GPU/FP16 support and wandb logging.
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "whisper-diarization-finetuned"),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        eval_strategy=config.get("eval_strategy", "epoch"),
        save_strategy=config.get("save_strategy", "epoch"),
        num_train_epochs=config.get("num_train_epochs", 1),
        learning_rate=config.get("learning_rate", 1e-4),
        fp16=config.get("fp16", True),
        logging_steps=config.get("logging_steps", 10),
        dataloader_num_workers=config.get("num_workers", 0),
        report_to="wandb",
        remove_unused_columns=False,
    )
    training_args.generation_config = peft_model.generation_config
    training_args.generation_max_length = 448

    # Instantiate the Seq2SeqTrainer.
    trainer = Seq2SeqTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        processing_class=processor,
    )
    trainer.label_names = ["labels"]

    # Log which device is used.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)

    # Begin training.
    trainer.train()
    trainer.save_model(str(CHECKPOINT_DIR / "whisper-diarization-finetuned"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()
    main(args.config)
