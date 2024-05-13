#!/bin/bash

sbatch jobs/inference_1.job
sbatch jobs/inference_2.job
sbatch jobs/inference_3.job
squeue