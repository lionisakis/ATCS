import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


def messages_to_input(data, tokenizer, model, name_of_message=0):
    input_ids_list = []
    if name_of_message == 0:
        for target_group, Sentence, Prior_Emotion, _, Possible_Emotions in list(
            zip(*data)
        ):
            # Compute the correctness score
            messages = [
                {
                    "role": "user",
                    "content": f"You are a {target_group}. Your prior emotion is {Prior_Emotion} while having the following event:\n"
                    f"{Sentence}\n"
                    "What is the main emotion you would feel while experiencing this event?"
                    "Provide your answer with only one word expressing your emotion and an explanation of that emotion."
                    f"Please generate the response in the form of a Python dictionary string with keys 'emotion' and 'explanation, where value of 'emotion' is one value out of {Possible_Emotions} that expresses an emotion and value of 'explanation' is a string that explanation why you choose that emotion."
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                    "For example, your response should look like this: {'emotion': 'Anger', 'explanation': 'I would feel anger when I had a serious argument with a dear person'.}.",
                }
            ]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            input_ids_list.append(input_ids)
    padded_input_ids, lengths = pad_sequence(input_ids_list, batch_first=True)
    return padded_input_ids, lengths


def annotate(prediction_set, output_dir, name_of_message):
    """
    Evaluates question and answer pairs using Llama-3
    Returns a score for correctness.
    """
    model_id = "../model_zoo/meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )

    batch_count = 0
    for data in tqdm(prediction_set):
        try:
            input_ids_list, lengths = messages_to_input(
                data, tokenizer, model, name_of_message
            )
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            input_ids_batch = torch.cat(input_ids_list, dim=0)
            packed_input = pack_padded_sequence(
                input_ids_batch, lengths, batch_first=True, enforce_sorted=False
            )
            outputs = model.generate(
                packed_input,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            # Convert response to a Python dictionary.
            padded_outputs, _ = pad_packed_sequence(outputs, batch_first=True)
            result_qa_pair = []
            print("len(outputs)", len(outputs))
            print("outputs.shape", outputs.shape)
            for length, output in zip(lengths, padded_outputs):
                response = output[length:]  # Trim padding
                response_message = tokenizer.decode(response, skip_special_tokens=True)
                response_dict = ast.literal_eval(response_message)
                result_qa_pair.append(response_dict)

            print(result_qa_pair)
            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{batch_count}.json", "w") as f:
                json.dump(result_qa_pair, f)

            batch_count += 1
        except Exception as e:
            print(f"Error processing file '{batch_count}': {e}")
