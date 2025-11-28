## Hugging Face Space

[Hugging Face Space](https://huggingface.co/spaces/Antuke/FaR-FT-PE)

[Demo Repo](https://github.com/Antuke/demo_far)

## Setup

Before you can execute the scripts provided, please run the following command, while in the folder that contains the file "setup.py":
pip install -r requirements.txt (if the device has gpu, follow pytorch instruction to have torch enabled with gpu)
pip install -e .

Also the dataset provided are zipped and need to be unzipped. The dataset_with_standard_labels folder should look like this:
```
datasets_with_standard_labels/
├── FairFace/
│   ├── train/
│   │   └── images_336/
│   │       ├── 0000.jpg
│   │       ├── 0001.jpg
│   │       └── ...
│   └── test/
│       └── images_336/
│           ├── 0030.jpg
│           ├── 0040.jpg
│           └── ...
└── RAF-DB/
    ├── train/
    │   └── ...
    └── test/
        └── ...
```

(Raccomended to at least unzip FairFace and RAF-DB, so to be able to test each task properly)

## Quick Start 
There is a unified entry point `main.py` and YAML configuration files for easier usage.

### Training
To start training using the default configuration:
```bash
python main.py train --config configs/train_config.yaml
```
You can modify `configs/train_config.yaml` to change parameters like `dataset_names`, `tasks`, `epochs`, etc.
You can also override parameters from the command line:
```bash
python main.py train --config configs/train_config.yaml --epochs 50 --batch_size 32
```

### Testing
To run tests:
```bash
python main.py test --config configs/test_config.yaml
```
Make sure to update `configs/test_config.yaml` with the correct `ckpt_path` (checkpoint path) before running.

### Demo
To run the demo:
```bash
python main.py demo --config configs/demo_config.yaml
```

## Legacy Scripts
Alternative, you can use the scripts in the `scripts/` folder. See below for instructions.

## Launch Demo
To launch a demo, you can execute the following command:
sh scripts/run_demo.sh 

By default, this script will instanciate the DoRA MTL trained model, and classify each image in the demo_images folder.
Result will be saved in the predicted folder.

## Launch Test(s)
To launch the test, be in the directory where "readme.md" file is placed (this file) as the scripts use relative paths.
Then you can launch the following scripts, as:

```
sh scripts/run_test_{task}_dora.sh         	<--- Test with DoRA modules
sh scripts/run_test_{task}_deep_head.sh    	<--- Test with deep classification head
sh scripts/run_test_{task}_attention.sh    	<--- Test with attention tuning
sh scripts/run_test_{task}_4unfreeze.sh    	<--- Test with fine-tune of the last 4 layers (+attention layer)
sh scripts/run_test_mtl_dora.sh 		    <--- Test with MTL DoRA
sh scripts/run_test_mtl_mtlora.sh 		    <--- Test with MTLoRA + DoRA
sh scripts/run_test_mtl_4unfreeze.sh 		<--- Test with fine-tune of the last 4 layers (+attention layer) for MTL
sh scripts/run_test_mtl_dora_mtlora4.sh  	<--- Test with MTLoRA + DoRA on last 4 layers
```

Substitute {task} with one of the following: "gender", "age" or "emotion".
So if you want to run the test scripts for DoRA modules trained for the emotion task you would run the following command:
sh scripts/run_test_emotion_dora.sh  

Once the testing is finished, the result can be found in the "testing_outputs" folder. Each test produces a new folder in there, containing:
- confusion_matrices folder, that contains .png of normalized and non-normalized confusion matrices per test-set
- test_result.json, that reports the accuracy and balanced accuracy obtained on each test-set.

The run_test_age_* 	    scripts will run a test on the "FairFace" Dataset by default  
The run_test_emotion_* 	scripts will run a test on the "RAF-DB" Dataset by default  
The run_test_gender_* 	scripts will run a test on the "FairFace" Dataset by default  
The run_mtl_*           scripts will run a test on the "FairFace" and "RAF-DB" dataset by default


If you want to run test on different dataset, open the corrisponding script that you want to run, and modify the DATASET_NAMES field.
For example if you want to run a run_test_age_* script and test on UTK and Vgg you can modify the field as such:
DATASET_NAMES="UTK Vgg"

All the dataset are provided with face crop already applied.

All the test results that are reported in the thesis were obtained with following hardware/software configuration:
```
GPU                    NVIDIA L40S  
CUDA Version           12.5  
Driver Version         555.42.06 
torch                  2.5.1
torchaudio             2.5.1
torchcodec             0.8.0
torchdata              0.11.0
torchvision            0.20.1
triton                 3.1.0
```

## Produce PCA or t-SNE graphs
To launch one of the two visualization method, you can run one of the following commands:
```
sh scripts/produce_tsne_graph.sh       
sh scripts/produce_pca_graph.sh 
```
By the default, they will both instanciate the DoRA MTL model, and produce the respective graph for the emotion task on the RAF-DB Dataset.
Check the .sh file if you want to change this default configurations


## Launch Train(s)
To train, you can check out the file "run_train.sh" , placed in the scripts folder.
Then, you can run the following command:
```
sh scripts/run_train.sh
```
After the training is finished, the model will be saved in the "training_outputs" folder.
To test it, you can check out the file "run_test.sh" , placed in the scripts folder.
Then, you can run the following command:
```
sh scripts/run_test.sh
```
After the testing is finished, the result can be found in the "testing_outputs" folder, where you will find:
- confusion_matrices folder, that contains .png of normalized and non-normalized confusion matrices per test-set
- test_result.json, that reports the accuracy and balanced accuracy obtained on each test-set.

## Acknowledgements
This project contains code from the Perception Encoder repository (https://github.com/facebookresearch/perception_models), licensed under Apache License, Version 2.0, January 2004
And code from the MTLoRA repository (https://github.com/scale-lab/MTLoRA), licensed under the MIT License. 
