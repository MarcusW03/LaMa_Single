## This code was sourced from [https://github.com/advimman/lama](https://github.com/advimman/lama). 

### This version has been modified from the original source. 

### Specifically, bin/predict_single.py includes a wrapper class for the model, which streamlines the inpainting process for a single image and mask. 

### Example usage in example_usage.py file


## To Use This Repository
- Download model checkpoint from [here](https://drive.google.com/file/d/11RbsVSav3O-fReBsPHBE1nn8kcFIMnKp/view?usp=drive_link) 
- Move best.ckpt to assets/lama_checkpoint folder. File path should be assets/lama_checkpoint/best.ckpt
- Create virtual environment. *See next section for details*
- Activate virtual environment. 
- Run example_usage.py or your own implementation. 

# Using a Virtual Environment
Running this repository locally requires a virtual environment. If you have one great, just run 'pip install -r requirements.txt' inside your environment. 

## Setting up an environment
- ### Using Python
  You can create a python virtual environment by running 'python -m venv .venv' in your project directory.
  You can activate the environment by running 'source ./.venv/bin/activate'.
- ### Using Conda (pre-requisite having conda installed)
  You can create a conda virtual environment by running 'conda create --name env'.
  You can activate the envionment by running 'conda activate env'
- ### Installing Requirements
  Once you have a running environment, you can run 'pip install -r requirements.txt' to install the required libraries.



#### Modifed by Marcus Wright
