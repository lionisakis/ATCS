import os
import json
import pandas as pd
from torch.utils.data import Dataset

preproceses_data_enlSEAR = {
    "D": "days_ago",
    "W": "weeks_ago",
    "M": "months_ago",
    "Y": "years_ago",
    "N": "not_very",
    "Mi": "moderately_intense",
    "I": "intense",
    "Vi": "very_intense",
    "Fm": "a_few_minutes",
    "H": "an_hour",
    "Sh": "several_hours",
    "Dom": "a_day_or_more",
    "Ml": "male",
    "Fl": "female",
    "O": "other",
    "GBR": "United Kingdom",
    "IRL": "Ireland",
}


class CustomDataset(Dataset):
    def __init__(
        self,
        tsv_file,
        target_group,
        Possible_Emotions=[
            "Anger",
            "Disgust",
            "Fear",
            "Guilt",
            "Joy",
            "Sadness",
            "Shame",
        ],
    ):
        self.data = pd.read_csv(tsv_file, sep="\t")
        # self.data["Target_Group"] = target_group
        self.list_of_dicts = self.data.to_dict("records")
        self.sentence_id = self.data["Sentence_id"].values
        # print(self.sentence_id)

        # self.Possible_Emotions= Possible_Emotions
        # self.target_group= self.data["Target_Group"].values
        # self.Sentence = self.data["Sentence"].values
        # self.Prior_Emotion = self.data["Prior_Emotion"].values
        # self.Annotation = self.data["Annotation"].values
        # self.City = self.data["City"].values
        # self.Temporal_Distance = self.preprocess(self.data['Temporal_Distance'].values) if 'Temporal_Distance' in self.data else []
        # self.Intensity = self.preprocess(self.data['Intensity'].values) if 'Intensity' in self.data else []
        # self.Duration = self.preprocess(self.data['Duration'].values) if 'Duration' in self.data else []
        # self.Gender = self.preprocess(self.data['Gender'].values) if 'Gender' in self.data else []
        # self.Country = self.preprocess(self.data['Country'].values) if 'Country' in self.data else []

        # todo
        # 1. discard other gender

    def preprocess(self, values):
        new_values = []
        for i in values:
            if i in preproceses_data_enlSEAR:
                new_values.append(preproceses_data_enlSEAR[i])
            else:
                new_values.append(i)
        return new_values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            # self.target_group[idx],
            self.Sentence[idx],
            self.Prior_Emotion[idx],
            self.Annotation[idx],
            self.Possible_Emotions,
        )
