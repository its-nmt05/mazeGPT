from maze_dataset import (
    read_from_file, 
    create_dataset, 
    save_to_file, 
    check_valid_path)
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from utils import parse_llm_res
from datasets import Dataset
import torch
from tqdm import tqdm


def build_prompt(maze, include_solution=False):
    prompt = f"""Solve this maze by finding a path from S to E.

Maze:
{maze['maze']}

Legend:
- S = Start position at {maze['start']}
- E = End position at {maze['end']}
- # = Wall (cannot pass through)
- Empty spaces = Valid paths

Rules:
- You can only move up, down, left, or right (no diagonal moves)
- You cannot pass through walls (#)
- Provide the solution as a list of coordinates

Output format: [[row, col], [row, col], ...]

Solution:"""

    if include_solution:
        shortest_path = min(maze['paths'], key=len) if maze['paths'] else []
        return prompt + f" {shortest_path}"
    
    return prompt


def prep_sft_dataset(maze_data, tokenizer):
    formatted_data = []
    
    for maze in tqdm(maze_data, desc="tokenizing dataset: "):
        prompt = build_prompt(maze, include_solution=False)
        full_text = build_prompt(maze, include_solution=True)

        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding=False
        )["input_ids"]

        full_tok = tokenizer(
            full_text,
            truncation=True,
            max_length=512,
            padding=False
        )

        input_ids = full_tok["input_ids"]
        attention_mask = full_tok["attention_mask"]
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        labels = labels[:len(input_ids)]
        
        formatted_data.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        })
    
    return Dataset.from_list(formatted_data)


def train(
    model_name="gpt2-large",
    train_data=None,
    test_data=None,
    output_dir="./sft_model",
    num_epochs=3,
    batch_size=2,
    learning_rate=3e-4,
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" 
    model.config.pad_token_id = tokenizer.eos_token_id

    # LoRA finetuning
    if use_lora:
        print("Configuring LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj",
                "o_proj", "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        print(f"Total params: {(sum(p.numel() for p in model.parameters())/1e6):.2f}M")
        model.print_trainable_parameters()

    print("Preparing datasets...")
    train_dataset = prep_sft_dataset(train_data, tokenizer)
    test_dataset = prep_sft_dataset(test_data, tokenizer)

    # auto-pad to max length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # no masked LM
        pad_to_multiple_of=8 
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_steps=500,
        eval_steps=500 if test_dataset else None,
        eval_strategy="steps",
        load_best_model_at_end=True,
        warmup_steps=100,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    if use_lora:
        model.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer


def eval_sft_model(model, tokenizer, eval_data):
    model.eval()
    correct = 0
    total = 0

    for maze in tqdm(eval_data, desc="Evaluating..."):
        prompt = build_prompt(maze, include_solution=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            # sample from LLM
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=False,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        gen = outputs[0][inputs["input_ids"].shape[-1]:]
        res = tokenizer.decode(gen, skip_special_tokens=True)
        path = parse_llm_res(res)
    
        if len(path) == 0:
            continue
        elif check_valid_path(maze, path):
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


def main():
    config = {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "n_tot": 10000,
        "n_train": 9000,
        "n_test": 1000,
        "num_epochs": 1,
        "batch_size": 1,
        "lr": 3e-4,
        "output_dir": "./sft_maze_model"
    }

    try:
        dataset = read_from_file('./maze_data.json')
    except Exception:
        dataset = create_dataset(num=config["n_tot"])
        save_to_file(dataset)

    train_data = dataset[:config["n_train"]]
    test_data = dataset[config["n_train"]:config["n_train"] + config["n_test"]]

    trained_model, tokenizer = train(
        model_name=config["model_name"],
        train_data=train_data,
        test_data=test_data,
        output_dir=config["output_dir"],
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["lr"],
    )
    
    accuracy = eval_sft_model(trained_model, tokenizer, test_data)
    print(f"Test Accuracy: {accuracy:.2f}")


def test():
    test_dataset = read_from_file('./maze_data.json')[9000:]
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    lora_path = "/teamspace/studios/this_studio/mazeGPT/sft_maze_model/checkpoint-9000"

    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model.eval()

    eval_sft_model(model, tokenizer, test_dataset)


if __name__ == "__main__":
    test()