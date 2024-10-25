import numpy as np

import os
os.system("mkdir Scripts")

for Bias in ["Forward", "Reverse"]:
    for D in ["2", "4"]:
        for E in  np.arange(0,5,0.2):
            E_value = str(E)[0:3]

            lines = []

            lines.append("#!/bin/bash")
            lines.append(f"#SBATCH --job-name=Heat_Rect_Mesos_{Bias}_D={D}_E={E_value}")
            lines.append("#SBATCH --gres=gpu:1")
            lines.append("#SBATCH --clusters=gpu")
#             lines.append("#SBATCH --partition=gtx1080")
            lines.append("#SBATCH --partition=a100")
            lines.append("#SBATCH --nodes=1")
            lines.append("#SBATCH --cores=1") #For finding the thermal state we need CPU, everything else can be done in the GPU
            lines.append("#SBATCH --time=0-1:00:00") #1 Hour. It should not take more than one hour.
            lines.append("#SBATCH --qos=short") 
            lines.append("#SBATCH --mail-user=jop204@pitt.edu") 
            lines.append("#SBATCH --mail-type=END,FAIL") 
            lines.append(f"#SBATCH --output=/ihome/jmendoza-arenas/jop204/Outputs/Heat_Rect_Mesos_{Bias}_D={D}_E={E_value}.out")
            lines.append(f"julia --threads=1 /ihome/jmendoza-arenas/jop204/Heat_Rectification_in_Tilted_Systems/Main_Script.jl {Bias} {D} {E_value}")
            
            script = open (f'Scripts/Heat_Rect_Mesos_{Bias}_D={D}_E={E_value}.slurm','w')
            script.write(lines[0])
            for i in range(1,len(lines)): script.write('\n' + lines[i])
            script.close()
            
#             os.system(f"sbatch Scripts/Heat_Rect_Mesos_{Bias}_D={D}_E={E_value}.slurm")
            
os.system("sbatch Scripts/Heat_Rect_Mesos_Forward_D=2_E=0.0.slurm")
os.system('rm -rf Scripts')
