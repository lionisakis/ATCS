from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm
import argparse
from dataset import CustomDataset
import os
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="question-answer-generation-using-gpt-3"
    )
    parser.add_argument(
        "--model_name",
        default="model_zoo/meta-llama/suzume-llama-3-8B-multilingual",
        type=str,
        help="What the name of the model that will be imported by hagging face.",
    )
    parser.add_argument(
        "--target_group", default="female", type=str, help="What is your target group."
    )
    parser.add_argument(
        "--name_of_message", default=0, type=int, help="what prompt type should use"
    )
    parser.add_argument(
        "--template", default=1, type=int, help="what template format should we use"
    )
    parser.add_argument(
        "--pred_path",
        default="dataset/deISEARenISEAR/de2enISEAR.tsv",
        help="The path to file containing prediction.",
    )
    parser.add_argument(
        "--output_dir",
        default="./suzame_output",
        help="The path to save annotation json files.",
    )
    parser.add_argument(
        "--batch_size", default=4, type=int, help="Number of batch sizes."
    )
    args = parser.parse_args()
    return args


class HFModel:
    def __init__(self, name, batch_size, new_tokens, target_group) -> None:
        self.print_message = True
        self.name = name
        self.target_group = target_group
        self.model = pipeline(
            "text-generation",
            model=name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if not self.model.tokenizer.pad_token_id:
            self.model.tokenizer.pad_token_id = self.model.model.config.eos_token_id

        self.batch_size = batch_size
        self.n_tokens = new_tokens

    def process(self, data, template):
        if template == 1:
            template = f"Adopt the identity of {self.target_group}. Answer the questions while staying in strict accordance with the nature of this identity. You went through the following event: "
        elif template == 2:
            template = f"Take the role of {self.target_group}. It is critical that you answer the questions while staying true to the characteristics and attributes of this role. You went through the following event: "
        else:
            template = f"You are a/an {self.target_group}. Your responses should closely mirror the knowledge and abilities of this persona. You went through the following event: "

        message = {
            "role": "user",
            "content": template + f"{data['Sentence']} "
            "What is the main emotion you would feel while experiencing this event? "
            "Provide your answer with only one word expressing your emotion and an explanation of that emotion. "
            "Please generate the response in the form of a Python dictionary string with keys 'emotion' and 'explanation', where value of 'emotion' is an emotion that expresses an emotion and value of 'explanation' is a string that explanation why you choose that emotion. "
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            'For example, your response should look like this: {"emotion": "Anger", "explanation": "I would feel anger when I had a serious argument with a dear person."}.',
        }
        if self.print_message:
            print(
                "A sample message the model took as input is the following:\n", message
            )
            self.print_message = False

        prompt = self.model.tokenizer.apply_chat_template([message], tokenize=False,)
        return prompt

    def generate_responses(self, dataset, template):
        sentence_id = dataset.sentence_id
        new_dataset = list(
            map(self.process, dataset.list_of_dicts, [template] * len(dataset))
        )
        responses = []
        # for i,response in tqdm(
        #     enumerate(
        #         self.model(
        #             new_dataset,
        #             batch_size=self.batch_size,
        #             max_new_tokens=self.n_tokens,
        #             do_sample=False,
        #             num_beams=1
        #             return_full_text=False,
        #         )
        #     ),
        #     total=len(new_dataset),
        # ):
        generated_responses = self.model(
            new_dataset,
            batch_size=self.batch_size,
            max_new_tokens=self.n_tokens,
            do_sample=False,
            num_beams=1,
            return_full_text=False,
        )
        for response in generated_responses:
            responses.append(
                response[0]["generated_text"]
                .replace("assistant\n\n", "")
                .replace("}", ', "target_group" : "' + self.target_group + '"}')
            )
        # responses.append(response[0]["generated_text"].replace("assistant\n\n", "").replace("}",', "target_group" : "'+self.target_group+'"}'))
        responses = {
            str(id_): text.replace(
                '"}', '", "event":"' + event["Sentence"].replace('"', "'") + '"}'
            )
            for id_, text, event in zip(sentence_id, responses, dataset.list_of_dicts)
        }
        print(responses)
        return responses


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    model = HFModel(
        name=args.model_name,
        batch_size=args.batch_size,
        new_tokens=256,
        target_group=args.target_group,
    )

    # Load data
    dataset = CustomDataset(args.pred_path, args.target_group)
    responses = model.generate_responses(dataset, args.template)

    output_dir = args.output_dir
    # Generate output directory if not exists.
    try:
        os.makedirs(output_dir, exist_ok=True)
        print("Directory created successfully")
    except OSError as error:
        print(f"Error creating directory: {error}")

    try:
        os.makedirs(output_dir + "/" + args.model_name.split("/")[-1], exist_ok=True)
        print("Directory created successfully")
    except OSError as error:
        print(f"Error creating directory: {error}")

    with open(
        output_dir
        + "/"
        + args.model_name.split("/")[-1]
        + "/"
        + args.target_group
        + "_"
        + str(args.template)
        + "_results.json",
        "w",
    ) as f:
        json.dump(responses, f)

    print("All evaluation completed!")


if __name__ == "__main__":
    main()
