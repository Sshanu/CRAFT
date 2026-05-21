import os
import json
from collections import Counter
import dirtyjson
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sentence_transformers
import re
import random
import string

def parse_markdown_sections(md_text):
    """
    Split md_text into sections at the shallowest heading level:
      1) Scan all headings to find the minimum depth.
      2) Split only at headings with that depth.
      3) Include any preamble as 'None' if content before the first heading.
      4) Return a list of {NodeID: heading, 'SubText': content} dicts.
    """
    lines = md_text.splitlines()
    all_headings = [
        (i, len(m.group(1)), m.group(0).strip())
        for i, line in enumerate(lines)
        if (m := re.match(r'^(#+)\s*(.*)', line))
    ]
    depths = [d for _, d, _ in all_headings]
    if depths:
        shallowest = min(depths)
        split_points = [(i, text) for i, d, text in all_headings if d == shallowest]
    else:
        return _build_sections([(None, md_text.strip())])

    sections = []
    # Preamble if exists
    first_idx = split_points[0][0]
    if first_idx > 0:
        pre = "\n".join(lines[:first_idx]).strip()
        if pre:
            sections.append((None, pre))

    # Sections for each shallow heading
    for idx, (line_no, heading_text) in enumerate(split_points):
        start = line_no + 1
        end = split_points[idx + 1][0] if idx + 1 < len(split_points) else len(lines)
        sub = "\n".join(lines[start:end]).strip()
        sections.append((heading_text, sub))

    return _build_sections(sections)

def _build_sections(pairs):
    used = set()
    def new_guid():
        chars = string.digits + string.ascii_uppercase
        while True:
            g = "".join(random.choices(chars, k=3))
            if g not in used:
                used.add(g)
                return g

    result = []
    for order, (hd, txt) in enumerate(pairs):
        guid = new_guid()
        node_id = f"H-1-{order}-{guid}"
        result.append({
            node_id: hd if hd is not None else "None",
            "SubText": txt
        })
    return result

def plot_metrics(eval_round_numbers, roundwise_report, output_dir, task_name):
    # Plot metrics
    avg_metrics = {}
    max_metrics = {}
    for col in roundwise_report.columns:
        if col.endswith('_avg'):
            metric = col[:-4]
            avg_metrics[metric] = roundwise_report[col].tolist()
        elif col.endswith('_max'):
            metric = col[:-4]
            max_metrics[metric] = roundwise_report[col].tolist()

    for metric, avg_metric in avg_metrics.items():
        plt.figure(figsize=(10, 5))
        plt.plot(eval_round_numbers, avg_metric, marker='o', label=f'Avg {metric}')
        plt.plot(eval_round_numbers, max_metrics[metric], marker='s', label=f'Max {metric}')
        plt.xlabel('Round')
        plt.ylabel(metric)
        plt.title(f"{task_name} - {metric} ")
        plt.legend()
        plot_filepath = os.path.join(output_dir, f"{metric}.png")
        plt.savefig(plot_filepath, bbox_inches="tight")
        plt.close()

def dynamic_similarity_threshold(embeddings,
                                  min_threshold: float = 0.80,
                                  percentile: int = 90) -> float:
    """
    Estimate an appropriate cosine‑similarity threshold from the current batch.

    • Computes all pair‑wise cosine similarities (upper‑triangle, excluding self‑pairs).  
    • Takes the chosen percentile (e.g. 90th) as the cut‑off so that only the
      top‑X % of the most‑similar pairs will be considered “same cluster”.  
    • Guarantees the threshold is **never below `min_threshold`**.

    Returns
    -------
    float
        The threshold to use for clustering.
    """
    if len(embeddings) < 2:        # nothing to compare
        return min_threshold

    # Collect upper‑triangle similarities into a list
    sims = []
    for i in range(len(embeddings)):
        cos_row = sentence_transformers.util.cos_sim(embeddings[i], embeddings[i+1:]).flatten()
        sims.extend(cos_row.tolist())

    if not sims:                   # single element after slicing
        return min_threshold

    dyn_thr = np.percentile(sims, percentile)
    return max(dyn_thr, min_threshold)

def flatten_json(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

class TokenUsage:
    def __init__(self, tokens={}):
        if 'prompt_tokens' not in tokens:
            self.prompt_tokens = 0 
        else:
            self.prompt_tokens = tokens['prompt_tokens']
        if 'completion_tokens' not in tokens:
            self.completion_tokens = 0
        else:
            self.completion_tokens = tokens['completion_tokens']
        if 'total_tokens' not in tokens:
            self.total_tokens = 0
        else:
            self.total_tokens = tokens['total_tokens']
    
    def update_token_usage(self, tokens):
        if tokens:
            self.prompt_tokens += tokens.prompt_tokens
            self.completion_tokens += tokens.completion_tokens
            self.total_tokens += tokens.total_tokens
        else:
            print(tokens)

class TokenUsageDetails:
    def __init__(self):
        self.token_usage_details = {}
        
    def init_components(self, components: list[str]):
        for component in components:
            self.token_usage_details[component] = TokenUsage({'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})

    def add_token_usage_for_component(self, token_usage, component_name: str, add=True):
        if add and component_name in self.token_usage_details:
            self.token_usage_details[component_name].update_token_usage(token_usage)
        else:
            self.token_usage_details[component_name] = token_usage
    
    def get_token_usage(self, component_name = None):
        if component_name is not None and component_name in self.token_usage_details:
            return self.token_usage_details.get(component_name)
        return TokenUsage()
    
    def calculate_overall_tokens(self):
        component_name = "overall"
        self.token_usage_details[component_name] = TokenUsage({'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
        for component in self.token_usage_details.values():
            self.token_usage_details[component_name].update_token_usage(component)

class Logger:
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def log(self, message, print_screen=True):
        if not isinstance(message, str):
            message = str(message)
        with open(self.out_dir, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
        if print_screen:
            print(message)
            
def contains_number(string):
    return any(char.isdigit() for char in string)

def get_keys_from_reference(reference):
    if reference[-1] == ">":
        reference = reference[:-1]
    keys = reference.split(">")
    keys = [key.strip() for key in keys]
    return keys

def is_leaf_node(key):
    if key in ['body', 'Examples']:
        return True
    return False

def clean_llm_response(llm_response: str) -> str:
    # Step 1: Handle fim suffix
    if "<|fim_suffix|>" in llm_response:
        llm_response = llm_response.split("<|fim_suffix|>")[0]

    text = llm_response.strip()

    # Step 2: Handle leading fence
    if text.startswith("```"):
        # remove the first fence
        text = text[3:].lstrip()
        # drop language specifier if present
        lines = text.splitlines()
        if lines and lines[0].strip().isalpha():
            lines = lines[1:]
        text = "\n".join(lines).strip()

    # Step 3: Handle trailing fence
    if text.endswith("```"):
        text = text[:-3].rstrip()

    return text.strip()

def extract_prompt_response(llm_response):
    # Remove all possible ending tags
    if "<|fim_suffix|>" in llm_response:
        llm_response = llm_response.split("<|fim_suffix|>")[0]
    if "<prompt_end>" in llm_response:
        llm_response = llm_response.split("<prompt_end>")[0]
    if "<compressed_prompt_end>" in llm_response:
        llm_response = llm_response.split("<compressed_prompt_end>")[0]
    if "<mutated_prompt_end>" in llm_response:
        llm_response = llm_response.split("<mutated_prompt_end>")[0]
    
    # Remove all possible starting tags
    if "<prompt_start>" in llm_response:
        llm_response = llm_response.split("<prompt_start>")[1]
    if "<compressed_prompt_start>" in llm_response:
        llm_response = llm_response.split("<compressed_prompt_start>")[1]
    if "<mutated_prompt_start>" in llm_response:
        llm_response = llm_response.split("<mutated_prompt_start>")[1]
        
    return llm_response.strip()

def find_headers_in_reference(references):
    found_headers = []
    for reference in references:
        keys = reference.split(">")
        found_headers.append(keys[0].strip())
    return found_headers

def clean_empty_dict(mapping):
    if not isinstance(mapping, dict):
        return mapping
    cleaned_dict = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            cleaned = clean_empty_dict(value)
            #value is empty dictionary so delete
            if len(cleaned.keys()) > 0:
                cleaned_dict[key] = cleaned
        elif value:
            cleaned_dict[key] = value
    return cleaned_dict

def att_to_dict(obj):
    """Recursively converts AttributeDict to a standard dictionary."""
    if isinstance(obj, dict):
        return {key: att_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [att_to_dict(item) for item in obj]
    else:
        return obj  # Return as is for non-dict, non-list values

def read_action_response(file_path):
    action_json = None
    with open(file_path, 'r', encoding='utf-8') as file:
        action_str = file.read() 
        json_part = action_str.split("```")[0].strip()
        try:
            action_json = att_to_dict(dirtyjson.loads(json_part))
        except Exception as e:
            print(f"Error {e} decoding JSON in file: {file_path}")
    return  action_json
                    
def read_action_response(file_path):
    """Reads and parses the action response JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def extract_action_types(directory_path):
    """Extracts and counts occurrences of action types from actor response JSON files."""
    action_types = []

    for filename in os.listdir(directory_path):
        if 'actor_response' in filename or 'exampleupdate_response' in filename:  
            file_path = os.path.join(directory_path, filename)
            action_json = read_action_response(file_path)

            if not action_json:
                continue  # Skip if file reading fails
            
            # Process a list of actions instead of a dictionary
            action_json = action_json["Actions"]
            if isinstance(action_json, list):
                for action in action_json:
                    action_type = action.get("Action")
                    is_example = action["Params"].get("IsExample", False)

                    if action_type:
                        if is_example:
                            action_types.append(f"Example-{action_type}")
                        else:
                            action_types.append(action_type)
            else:
                print(f"Unexpected format in {filename}: Expected a list of actions.")

    # Count the frequency of each action type
    return Counter(action_types)

def plot_actions(current_round, action_count, output_dir):
    labels = list(action_count.keys())
    values = list(action_count.values())
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color='royalblue')
    plt.xlabel('Action Types')
    plt.ylabel('Frequency')
    plt.title(f'Frequency of Action Types {current_round}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir + "/action_types.png")

def check_keys_with_period(section):
    keys_with_period = [key for key in section if '.' in key]
    return keys_with_period

def log_actions(current_round, output_dir, rounddir, action_types_counter):
    action_count = extract_action_types(rounddir)
    action_df = pd.DataFrame(action_count.items(), columns=['Action Type', 'Frequency'])
    action_df.to_csv(rounddir + "/action_types.tsv", sep='\t', index=False)
    plot_actions(current_round, action_count, rounddir)
    action_types_counter.update(action_count)

    overall_action_df = pd.DataFrame(action_types_counter.items(), columns=['Action Type', 'Frequency'])
    overall_action_df.to_csv(output_dir + "/action_types.tsv", sep='\t', index=False)