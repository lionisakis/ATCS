#!/bin/bash

# ENGLISH GENDER
# sbatch jobs/inference_gender_aya-101.job
# # sbatch jobs/inference_gender_Llama-2-7b-hf.job
# sbatch jobs/inference_gender_Meta-Llama-3-8B-Instruct.job
sbatch jobs/inference_gender_suzume-llama-3-8B-multilingual.job

# GERMAN GENDER
# sbatch jobs/inference_german_gender_aya-101.job
# # sbatch jobs/inference_german_gender_Llama-2-7b-hf.job
# sbatch jobs/inference_german_gender_Meta-Llama-3-8B-Instruct.job
sbatch jobs/inference_german_gender_suzume-llama-3-8B-multilingual.job

# ENGLISH RACE
# sbatch jobs/inference_race_aya-101.job
# # sbatch jobs/inference_race_Llama-2-7b-hf.job
# sbatch jobs/inference_race_Meta-Llama-3-8B-Instruct.job
sbatch jobs/inference_race_suzume-llama-3-8B-multilingual.job

# GERMAN RACE
# sbatch jobs/inference_german_race_aya-101.job
# # sbatch jobs/inference_german_race_Llama-2-7b-hf.job
# sbatch jobs/inference_german_race_Meta-Llama-3-8B-Instruct.job
sbatch jobs/inference_german_race_suzume-llama-3-8B-multilingual.job

# ENGLISH RACE GENDER
# sbatch jobs/inference_race_gender_aya-101.job
# # sbatch jobs/inference_race_gender_Llama-2-7b-hf.job
# sbatch jobs/inference_race_gender_Meta-Llama-3-8B-Instruct.job
sbatch jobs/inference_race_gender_suzume-llama-3-8B-multilingual.job

# GERMAN RACE GENDER
# sbatch jobs/inference_german_race_gender_aya-101.job
# # sbatch jobs/inference_german_race_gender_Llama-2-7b-hf.job
# sbatch jobs/inference_german_race_gender_Meta-Llama-3-8B-Instruct.job
sbatch jobs/inference_german_race_gender_suzume-llama-3-8B-multilingual.job

# RECALL
# # sbatch jobs/inference_recall_Meta-Llama-3-8B-Instruct.job
squeue
