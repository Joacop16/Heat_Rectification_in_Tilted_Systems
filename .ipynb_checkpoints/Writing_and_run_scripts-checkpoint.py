import numpy as np

import os
os.system("mkdir Scripts")

for D in [6, 8]:
    # for E in  np.arange(0.0,5,0.2):
    for E in  np.arange(0.0,10.0,0.2):
        for Bias in ["Forward", "Reverse"]:
            E_value = str(E)[0:3]

            lines = []

            lines.append("#!/bin/bash")
            lines.append(f"#SBATCH --job-name=Heat_Rect_Mesos_{Bias}_D={D}_E={E_value}")
            lines.append("#SBATCH --gres=gpu:1")
            lines.append("#SBATCH --clusters=gpu")
            lines.append("#SBATCH --partition=gtx1080") #11GB of GPU
#            lines.append("#SBATCH --partition=a100") #40 GB of GPU
            lines.append("#SBATCH --nodes=1")
            lines.append("#SBATCH --cores=1") #For finding the thermal state we need CPU, everything else can be done in the GPU
            lines.append(f"#SBATCH --time=0-{round(D/2)}:00:00") #2 Hour. It should not take more than one hour, but I put 2 just in case
            lines.append("#SBATCH --qos=short") 
            lines.append("#SBATCH --mail-user=jop204@pitt.edu") 
            lines.append("#SBATCH --mail-type=END,FAIL") 
            lines.append(f"#SBATCH --output=/ihome/jmendoza-arenas/jop204/Outputs/Heat_Rect_Mesos_{Bias}_D={D}_E={E_value}.out")
            lines.append(f"julia --threads=1 /ihome/jmendoza-arenas/jop204/Heat_Rectification_in_Tilted_Systems/Main_Script.jl {Bias} {D} {E_value}")
            
            script = open (f'Scripts/Heat_Rect_Mesos_{Bias}_D={D}_E={E_value}.slurm','w')
            script.write(lines[0])
            for i in range(1,len(lines)): script.write('\n' + lines[i])
            script.close()
            
            os.system(f"sbatch Scripts/Heat_Rect_Mesos_{Bias}_D={D}_E={E_value}.slurm")
            
# os.system("sbatch Scripts/Heat_Rect_Mesos_Reverse_D=4_E=0.0.slurm") #For testing
os.system('rm -rf Scripts')
