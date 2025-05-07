import json
import torch
import gc

from datasets import Dataset
from torch.utils.data import Dataset
from schema_item_filter import SchemaItemClassifierInference, filter_schema
from utils.db_utils import get_db_schema_sequence, get_matched_content_sequence
from string import Template

# INPUT_PROMPT = "You are given a SQLite database schema that includes table name, columns, data types and exemplar values. You are tasked to ask questions about the database by referencing the provided schema.\n"
TRAIN_ZERO_SHOT_PROMPT = "You are given a natural language question. You are tasked to answer the question by using a SQLite query. Do not include any other text in your response.\n"
EVAL_ZERO_SHOT_PROMPT_TEMPLATE = Template(open("utils/unified_prompt.txt", "r").read())

def prepare_text2sql_prefix_sequence(data, prompt_template = None) -> str:
    prefix_seq = ""
    if prompt_template is not None:
        prefix_seq = str(prompt_template.substitute(question=data["question"], 
                                                      schema_with_rows=data["schema_sequence"], 
                                                      external_knowledge=data["evidence"]))
    else:
        prefix_seq = data["schema_sequence"] + "\n" + data["content_sequence"] + "\n" + data["text"] + "\n"
    
    return prefix_seq

def prepare_text2sql_prefix_sequence_zero_shot(data):
    text = data["question"]

    return "\n" + text + "\n"

####################################################### this is for Qwen2.5
def prepare_inputs_and_labels_qwen(prefix_seq, target_seq, tokenizer, max_tokens):
    prefix_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq , truncation = False).input_ids
    prefix_ids = prefix_ids[1:]

    target_ids = tokenizer(target_seq, truncation = False).input_ids + [tokenizer.eos_token_id]
    seq_length = len(prefix_ids) + len(target_ids)

    # exclude bos_token_id for Qwen2.5 as it's None

    if seq_length <= max_tokens: # pad inputs with pad_token_id
        pad_length = max_tokens - seq_length
        input_ids = prefix_ids + target_ids + [tokenizer.pad_token_id] * pad_length
        # tell the model to ignore the padding tokens when performing (masked) self-attention 
        attention_mask = [1] * seq_length + [0] * pad_length
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids + [-100] * pad_length
    else: # no padding, meaning the input sequence exceeds the max_tokens
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = prefix_ids + target_ids
        # pre-truncate input ids
        # input_ids = [tokenizer.bos_token_id] + input_ids[-(max_tokens-1):]
        input_ids = input_ids[-max_tokens:]
        attention_mask = [1] * max_tokens
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids
        # pre-truncate labels
        labels = labels[-max_tokens:]
    
    return {
        "input_ids": torch.tensor(input_ids, dtype = torch.int64), 
        "attention_mask": torch.tensor(attention_mask, dtype = torch.int64), 
        "labels": torch.tensor(labels, dtype = torch.int64)
    }

def prepare_inputs_qwen(prefix_seq, tokenizer, max_prefix_length):
    input_ids = tokenizer(prefix_seq , truncation = False)["input_ids"] # exclude bos_token_id for Qwen2.5
    # input_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq , truncation = False)["input_ids"]

    if len(input_ids) > max_prefix_length:
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        # input_ids = [tokenizer.bos_token_id] + input_ids[-(max_prefix_length-1):]
        input_ids = input_ids[-max_prefix_length:] # exclude bos_token_id for Qwen2.5
    
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype = torch.int64), 
        "attention_mask": torch.tensor(attention_mask, dtype = torch.int64)
    }
#######################################################

def prepare_inputs_and_labels(prefix_seq, target_seq, tokenizer, max_tokens):
    prefix_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq , truncation = False)["input_ids"]
    target_ids = tokenizer(target_seq, truncation = False)["input_ids"] + [tokenizer.eos_token_id]

    seq_length = len(prefix_ids) + len(target_ids)
    if seq_length <= max_tokens: # pad inputs with pad_token_id
        pad_length = max_tokens - seq_length
        input_ids = prefix_ids + target_ids + [tokenizer.pad_token_id] * pad_length
        # tell the model to ignore the padding tokens when performing (masked) self-attention 
        attention_mask = [1] * seq_length + [0] * pad_length
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids + [-100] * pad_length
    else: # no padding
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = prefix_ids + target_ids
        # pre-truncate input ids
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_tokens-1):]
        attention_mask = [1] * max_tokens
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids
        # pre-truncate labels
        labels = labels[-max_tokens:]
    
    return {
        "input_ids": torch.tensor(input_ids, dtype = torch.int64), 
        "attention_mask": torch.tensor(attention_mask, dtype = torch.int64), 
        "labels": torch.tensor(labels, dtype = torch.int64)
    }

def prepare_inputs(prefix_seq, tokenizer, max_prefix_length):
    input_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq , truncation = False)["input_ids"]

    if len(input_ids) > max_prefix_length:
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_prefix_length-1):]
    
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype = torch.int64),
        "attention_mask": torch.tensor(attention_mask, dtype = torch.int64)
    }

class SFTSQLGenerationDataset(Dataset):
    def __init__(self, text2sql_data_dir, tokenizer, max_tokens, mode, table_num, column_num, sic_path, 
                 filter=True, target_key="sql", input_prompt="", seed=42):
        super().__init__()
        import numpy as np
        dataset = json.load(open(text2sql_data_dir))
        train_dataset = np.random.RandomState(seed).choice(dataset, size=int(len(dataset) * 0.8), replace=False)
        eval_dataset = np.setdiff1d(dataset, train_dataset)
        if mode == "train":
            dataset = train_dataset
        elif mode == "eval":
            dataset = eval_dataset
        else:
            raise ValueError(f"mode {mode} is not supported")
        print(f"==Loaded {len(dataset)} examples for {mode} set==")
        self.target_key = target_key
        self.input_prompt = input_prompt
        # print(f" skipping schemafiltering strategies...")
        # Let's not apply filtering strategies.
        
        print("apply filtering strategies...")
        if mode == "train":
            if filter:
                dataset = filter_schema(dataset, "train", None, table_num, column_num)
        elif mode == "eval":
            sic = SchemaItemClassifierInference(sic_path)
            if filter:
                dataset = filter_schema(dataset, "eval", sic, table_num, column_num)
            del sic
            torch.cuda.empty_cache()

        if "train-zero-shot" not in mode:
            # prepare schema sequence and content sequence
            for data in dataset:
                # get the schema and matched contents prompt
                data["schema_sequence"] = get_db_schema_sequence(data["schema"])
                data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])

        self.mode = mode
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __getitem__(self, index):
        data = self.dataset[index]
        prefix_seq = self.input_prompt + prepare_text2sql_prefix_sequence(data)

        if index < 2:
            print(prefix_seq)

        if "train" in self.mode:
            target_seq = data[self.target_key]
            return prepare_inputs_and_labels_qwen(prefix_seq, target_seq, self.tokenizer, self.max_tokens)
        elif "eval" in self.mode:
            return prepare_inputs_qwen(prefix_seq, self.tokenizer, self.max_tokens)
        else:
            raise ValueError(f"mode {self.mode} is not supported")

    def __len__(self):
        return len(self.dataset)