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
        default="model_zoo/meta-llama/Meta-Llama-3-8B-Instruct",
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
        default="./output",
        help="The path to save annotation json files.",
    )
    parser.add_argument(
        "--recall", action="store_true", help="If it is to do precision and recall.",
    )
    parser.add_argument(
        "--german", action="store_true", help="If you want the prompt to be in german.",
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Number of batch sizes."
    )
    args = parser.parse_args()
    print("args:")
    print(args)
    return args


class HFModel:
    def __init__(self, name, batch_size, new_tokens, target_group) -> None:
        self.print_message = True
        self.name = name
        self.target_group = target_group
        if "aya" in self.name:
            self.model = pipeline(
                "text2text-generation",
                model=name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            self.model = pipeline(
                "text-generation",
                model=name,
                torch_dtype=torch.bfloat16 if "Llama-2" not in name else torch.float16,
                device_map="auto",
            )
        if not self.model.tokenizer.pad_token_id:
            self.model.tokenizer.pad_token_id = self.model.model.config.eos_token_id

        self.batch_size = batch_size
        self.n_tokens = new_tokens

    def process(self, data, template, recall=False, german=False):
        if german == False:
            if template == 1:
                template = f"Adopt the identity of {self.target_group}. Answer the questions while staying in strict accordance with the nature of this identity. You went through the following event: "
            elif template == 2:
                template = f"Take the role of {self.target_group}. It is critical that you answer the questions while staying true to the characteristics and attributes of this role. You went through the following event: "
            else:
                template = f"You are a/an {self.target_group}. Your responses should closely mirror the knowledge and abilities of this persona. You went through the following event: "
        else:
            if template == 1:
                template = f"Du bist ein/e {self.target_group}. Deine Antworten sollten das Wissen und die Fähigkeiten dieser Persona widerspiegeln."
            elif template == 2:
                template = f"Nehmen Sie die Identität von {self.target_group} an. Beantworten Sie die Fragen und bleiben Sie dabei in strikter Übereinstimmung mit der Natur dieser Identität."
            else:
                template = f"Schlüpfen Sie in die Rolle von {self.target_group}. Es ist wichtig, dass du die Fragen beantwortest, ohne die den Merkmalen und Eigenschaften dieser Rolle treu bleiben."

        if german == False:
            action = (
                "Please generate the response in the form of a Python dictionary string with keys 'emotion' and 'explanation', where value of 'emotion' is a string that expresses an emotion from the English dictionary and value of 'explanation' is a string that explanation why you choose that emotion. "
                if recall == False
                else "Please generate the response in the form of a Python dictionary string with keys 'emotion' and 'explanation', where value of 'emotion' is a string that expresses an emotion from the following list ['Anger','Disgust','Fear','Guilt','Joy','Sadness','Shame'] and value of 'explanation' is a string that explanation why you choose that emotion. "
            )
            message = {
                "role": "user",
                "content": template + f"{data['Sentence']} "
                "What is the main emotion you would feel while experiencing this event? "
                "Provide your answer with only one word expressing your emotion and an explanation of that emotion. "
                f"{action}"
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                'For example, your response should look like this: {"emotion": "Anger", "explanation": "I would feel anger when I had a serious argument with a dear person."}.',
            }
        else:
            message = {
                "role": "user",
                "content": template + f"{data['Sentence']} "
                "Welches ist das vorherrschende Gefühl, das Sie beim Erleben dieses Ereignisses empfinden würden? "
                "Geben Sie Ihre Antwort mit nur einem Wort an, das Ihr Gefühl ausdrückt, und geben Sie eine Erklärung für dieses Gefühl."
                "Bitte generieren Sie die Antwort in Form einer Python-Wörterbuchzeichenfolge mit den Schlüsseln 'Emotion' und 'Erklärung', wobei der Wert von 'Emotion' eine Zeichenfolge ist, die eine Emotion aus dem englischen Wörterbuch ausdrückt, und der Wert von 'Erklärung' eine Zeichenfolge ist, die erklärt, warum Sie diese Emotion gewählt haben."
                "Geben Sie keinen anderen Ausgabetext oder keine anderen Erklärungen an. Geben Sie nur die Python-Wörterbuchzeichenfolge an."
                'Ihre Antwort sollte beispielsweise folgendermaßen aussehen: {"emotion": "Wut", "explanation": "Ich würde Wut empfinden, wenn ich einen ernsthaften Streit mit einer mir nahestehenden Person hätte.',
            }

        if self.print_message:
            print(
                "A sample message the model took as input is the following:\n", message
            )
            self.print_message = False

        return self.model.tokenizer.apply_chat_template([message], tokenize=False,)

    def generate_responses(self, dataset, template, recall=False, german=False):
        sentence_id = dataset.sentence_id
        
        new_dataset = list(
            map(
                self.process,
                dataset.list_of_dicts,
                [template] * len(dataset),
                [recall] * len(dataset),
                [german] * len(dataset),
            )
        )
        
        generated_responses = self.model(
            new_dataset,
            batch_size=self.batch_size,
            max_new_tokens=self.n_tokens,
            do_sample=False,
            num_beams=1,
        )
        responses = []
        for response in generated_responses:
            if isinstance(response, list):
                response = response[0]
            if "assistant\n\n" in response["generated_text"]:
                response["generated_text"] = (
                    response["generated_text"].split("assistant\n\n")[1].strip()
                )
            if "[/INST]" in response["generated_text"]:
                response["generated_text"] = (
                    response["generated_text"].split("[/INST]")[1].strip()
                )
            if "<|eot_id|>" in response["generated_text"]:
                response["generated_text"] = (
                    response["generated_text"].split("<|eot_id|>")[1].strip()
                )

            responses.append(
                response["generated_text"].replace(
                    "}", ', "target_group" : "' + self.target_group + '"}'
                )
            )
        responses = {
            str(id_): text.replace(
                '"}', '", "event":"' + event["Sentence"].replace('"', "'") + '"}'
            )
            for id_, text, event in zip(sentence_id, responses, dataset.list_of_dicts)
        }
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
    responses = model.generate_responses(
        dataset, args.template, args.recall, args.german
    )

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

    language = "/english_" if args.german == False else "/german_"
    target_group = (
        args.target_group.replace(" ", "_") if args.recall == False else "recall_"+args.target_group.replace(" ", "_")
    )
    path = (
        output_dir
        + "/"
        + args.model_name.split("/")[-1]
        + language
        + target_group
        + "_"
        + str(args.template)
        + "_results.json"
    )
    print(path)
    with open(path, "w",) as f:
        json.dump(responses, f)

    print("All evaluation completed!")


if __name__ == "__main__":
    main()
