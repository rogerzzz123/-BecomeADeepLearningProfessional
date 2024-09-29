# -BecomeADeepLearningProfessional
Educative Course


## - Conda Installation
https://docs.anaconda.com/anaconda/install/mac-os/

## - Create Virtual Environment
- conda create -n pytorchbook anaconda
- conda activate pytorchbook

" Done! You are using a brand new conda environment now. You’ll need to activate it every time you open a new terminal. Or if you’re a Windows or macOS user, you can open the corresponding Anaconda Prompt (it will show up as Anaconda Prompt (pytorchbook) in our case), which will have it activated from the start." 

## Install Pytorch

- conda install pytorch torchvision cpuonly -c pytorch
  
"CUDA “is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs).”"

macOS: If you’re a macOS user, please beware that PyTorch’s binaries do not support CUDA. Meaning, you’ll need to install PyTorch from source if you want to use your GPU. This is a somewhat complicated process. As described in this link, if you don’t feel like it, you can choose to proceed without CUDA, and you’ll still be able to execute the code in this course promptly.

简单说一句话： mac local就别用gpu了 非常复杂 哪里能用cuda就在那里用gpu

## Install TensorBoard
-  conda install -c conda-forge tensorboard

## Conda install vs Pip install

Although they may seem equivalent at first sight, you should prefer conda install over pip install when working with Anaconda and its virtual environments.

The reason is that conda install is sensitive to the active virtual environment; the package will be installed only for that environment. If you use pip install, and pip itself is not installed in the active environment, it will fall back to the global pip, and you definitely do not want that.

Why not? Remember the problem with dependencies that I mentioned in the virtual environment section? That’s why! The conda installer assumes it handles all packages that are part of its repository and keeps track of the complicated network of dependencies among them (to learn more about this, check this link).

## Start Jupyter Notebook in your Conda Env:
- jupyter notebook

## Running TensorBoard Locally
- tensorboard --logdir runs
- 

- 
