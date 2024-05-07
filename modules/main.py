
import os
import argparse
import json
from multiprocessing.pool import Pool
from dataset import CustomDataset
from torch.utils.data import DataLoader
from inferance import annotate

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--target_group", default='man',type=str, help="What is your target group.")
    parser.add_argument("--name_of_message", default=0, type=int, help="what prompt type should use")
    parser.add_argument("--pred_path", default='/home/scur0405/LLaMA-VID/ATCS/dataset/deISEARenISEAR/enISEAR_validation.tsv', help="The path to file containing prediction.")
    parser.add_argument("--output_dir", default='./output', help="The path to save annotation json files.")
    parser.add_argument("--output_json", default='./output/merged.json', help="The path to save annotation final combined json file.")
    parser.add_argument("--batch_size", default=64, type=int, help="Number of splits.")
    parser.add_argument("--num_chunks", default=1, type=int, help="Result splits")
    args = parser.parse_args()
    return args

def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    # Load prediction file
    dataset = CustomDataset(args.pred_path, args.target_group)
    evaluation = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    annotate(evaluation, output_dir,args.name_of_message)

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

if __name__ == "__main__":
    main()
