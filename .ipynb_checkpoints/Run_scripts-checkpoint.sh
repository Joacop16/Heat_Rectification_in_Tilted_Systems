#!/bin/bash

rm -rf Scripts
python Writing_and_run_scripts.py

sbatch Scripts/Heat_Rect_Mesos_Forward_D=2_E=0.0.slurm