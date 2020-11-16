#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J collect_data_ny
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=32GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 32GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s153900@student.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Error_%J.err

# -- end of LSF options --
# here follow the commands you want to execute 
module load python3
pip3 install --user --upgrade pip
pip3 install --user torch torchvision
module unload python3

module load cuda/10.2
nvidia-smi
/appl/cuda/10.2/samples/NVIDIA_CUDA-10.2_Samples/bin/x86_64/linux/release/deviceQuery

module load python
module load rdkit

pip install --user torch torchvision
cd JTVAE-on-Molecular-Structures/fast_molvae
python preprocess.py --train ../QM9/train.txt --split 100 --jobs 16
mkdir QM9-processed
mv tensor* QM9-processed

