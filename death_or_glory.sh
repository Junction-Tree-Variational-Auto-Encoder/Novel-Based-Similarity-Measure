#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J train_model
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

module load python3/3.7.7
python3 -m venv venv_1
source venv_1/bin/activate
bash Anaconda3-5.2.0-x86_64.sh
conda install -y cmake cairo pillow eigen pkg-config
conda install -y boost-cpp boost py-boost

conda install -y gxx_linux-64
git clone https://github.com/rdkit/rdkit.git
cd rdkit
mkdir build && cd build
cmake -DPy_ENABLE_SHARED=1 \
  -DRDK_INSTALL_INTREE=ON \
  -DRDK_INSTALL_STATIC_LIBS=OFF \
  -DRDK_BUILD_CPP_TESTS=ON \
  -DPYTHON_NUMPY_INCLUDE_PATH="$(python -c 'import numpy ; print(numpy.get_include())')" \
  -DBOOST_ROOT="$CONDA_PREFIX" \
  ..

module load cuda/10.2
nvidia-smi
/appl/cuda/10.2/samples/NVIDIA_CUDA-10.2_Samples/bin/x86_64/linux/release/deviceQuery

pip install torch===1.7.0 torchvision===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
module load pandas/1.0.3-python-3.7.7
module load scipy/1.4.1-python-3.7.7
module load numpy/1.18.2-python-3.7.7-openblas-0.3.9
pip install --user tqdm
pip3 install --user tqdm

python3 main_train.py
