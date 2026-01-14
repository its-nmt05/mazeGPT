from maze_dataset import create_dataset, save_to_file, read_from_file, check_valid_path
import re
import ast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def format_path(path):
    directions = []
    for i in range(len(path) - 1):
        curr_pos = path[i]
        next_pos = path[i+1]

        row_diff = next_pos[0] - curr_pos[0]
        col_diff = next_pos[1] - curr_pos[1]

        if row_diff == 1 and col_diff == 0:
            directions.append("down")
        elif row_diff == -1 and col_diff == 0:
            directions.append("up")
        elif row_diff == 0 and col_diff == 1:
            directions.append("right")
        elif row_diff == 0 and col_diff == -1:
            directions.append("left")
        else:
            raise ValueError(f"Invalid move from {curr_pos} to {next_pos}")
        
    return directions


def create_prompt(maze):
    prompt = f"""Solve this maze by finding a path from S to E.

    Maze:\n
    {maze['maze']}

    Legend:
    - S = Start position
    - E = End position  
    - # = Wall (cannot pass through)

    Rules:
    - You can only move up, down, left, or right (no diagonal moves)
    - You cannot pass through walls (#)
    - Provide the solution as a list of coordinates

    Output format:
    [[row, col], [row, col], ...]

    For example: [[0, 0], [0, 1], [1, 1], [1, 2]]

    Solution:"""

    return prompt


def load_llm_model(model_name='microsoft/Phi-3-mini-128k-instruct'):
    torch.random.manual_seed(0) 
    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    model = AutoModelForCausalLM.from_pretrained( 
        model_name,  
        device_map="cuda",  
        torch_dtype=torch.bfloat16,  
        trust_remote_code=True,  
    ) 
    return model, tokenizer


def generate_llm_res(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        top_p=0.9
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def evaluate_response(maze, llm_response):
    paths = re.findall(r'\[\s*\[\s*\d+\s*,\s*\d+\s*\](?:\s*,\s*\[\s*\d+\s*,\s*\d+\s*\])*\s*\]', llm_response)
    paths = [ast.literal_eval(p) for p in paths]

    for path in reversed(paths):
        if check_valid_path(maze, path):
            return True
        
    return False

    

# def format_dataset(maze_data):
#     dataset = []
#     for maze in maze_data:
#         path = " ".join(format_path(maze['path']))
#         res = f"""Input format:\n{maze['maze']}\n\nOutput:\n{path}"""
#         dataset.append(res)

#     return dataset


training_data = create_dataset(num=10, size_min=3, size_max=6)
save_to_file(training_data)
# data = read_from_file()
# prompt = create_prompt(data[0])
# print(prompt)

# model, tokenizer = load_llm_model("gpt2-medium")
# response = generate_llm_res(model, tokenizer, prompt)
# print(response)