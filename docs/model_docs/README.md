before run our project, you need first clone it to your local server.

```bash
git clone git@github.com:SEU-zxj/LLM-Augmented-MTR.git
```

# 1. Dataset Preparation

At first, you need to download the [Waymo Open Motion Dataset (WOMD)](https://waymo.com/open/download/) and then preprocess the dataset.

For more details, you can refer to [MTR's Data Preparation](https://github.com/sshaoshuai/MTR/blob/master/docs/DATASET_PREPARATION.md). (MTR use `waymo-open-dataset-tf-2-6-0`, but we choose `waymo-open-dataset-tf-2-11-0`. You can change it or not.)

After you prepared WOMD, our context data should be downloaded next. As these files are huge, so we upload it to Google Drive, you can download then via this [link](https://drive.google.com/drive/folders/16mVoB5Su7IZ3WHxLVS-yE7_Wh3eFG4zb?usp=sharing).

Just make sure these files are listed in the correct folder:

```bash
llm_augmented_mtr
├── data
|   └── ...
├── LLM_integrate
│   ├── context_data
│   │   ├── test
│   │   │   └── context_data_encoder_100.pkl
│   │   ├── train
│   │   │   └── context_data_encoder_100.pkl
│   │   └── valid
│   │       └── context_data_encoder_100.pkl
│   ├── embedding
```

**What are these context data?** These context data serve as extra information provided by LLM for the origin MTR, which can improve performance of origin MTR.

# 2. Install

**Step 1.** Create a virtual environment of python

```bash
conda create --name llm_augmented_mtr python=3.8
conda activate llm_augmented_mtr
```

**Step 2.** Install all the required packages

```bash
pip install -r requirements.txt
```

if you **failed** in this step, try to install packages below manually:

```bash
numpy
torch>=1.1
tensorboardX
easydict
pyyaml
scikit-image
tqdm
```

**Step 3.** Compile this codebase (As author of MTR has write serveral code of CUDA)

```bash
python setup.py develop
```

# 3. Train and Eval

Before train and eval our model, first make sure you set the right `DATA_DIR` in configuration files in `LLM_integrate/tools/cfgs` and `tools/cfgs`.

```yaml
DATA_CONFIG:
    DATASET: WaymoDataset
    OBJECT_TYPE: &object_type ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']
    DATA_ROOT: 'data/waymo/mtr_processed'
    # ^ Make sure you set the right root dir of WOMD
```

## 3.1 Training

Train our model is pretty easy.

```bash
cd tools
source run/dist_train_llm_augmented_mtr.sh
```

## 3.2 Eval one checkpoint on the validation set

Eval the performance of trained model is also simple.

```bash
# make sure your working directory is `llm_augmented_mtr/tools` 
# before run eval task, set the checkpoint path (--ckpt) in `run/dist_valid_llm_augmented_mtr.sh`
source run/dist_valid_llm_augmented_mtr.sh
```

## 3.3 Genertae result on testing set

If you want to generate result of the testing set, first make sure you have trained model and obtain cheackpoint.

```bash
# make sure your working directory is `llm_augmented_mtr/tools` 
# before run test task, set the checkpoint path (--ckpt) in `run/dist_test_llm_augmented_mtr.sh`
source run/dist_test_llm_augmented_mtr.sh
```

As WOMD do not provide ground truth on testing set, the final evaluation metrics would not show at last.

If you want to know the performance of your model, you need to submit your result file to WOMD server (the next step).

# 4. Submit to WOMD LeaderBoard

We also provide a format covert script in the `submit_to_waymo` folder, you can submit your model's result to [WOMD leaderboard](https://waymo.com/open/challenges/2024/motion-prediction/) (this link brings you to leaderboard of WOMD challenge 2024).

if you want to use this script, see intructions inside it and you can easily convert result data to submission version.

# 5. More Details

## 5.1 I want to run origin MTR

LLM-augmented is a plug-and-play module, you can set it or not. We provide execute scripts for users who want to train/eval the original MTR.

Before excute these task, change parameters inside the corresponding script (GPU, configuration file, batch size, tag).

The default parameter are set for MTR train/eval/test on the whole WOMD, if you want to train/eval/test on 20% WOMD, set `--cfg_file` to `cfgs/waymo/mtr+20_percent_data.yaml`, and `--extra_tag` to your liked tag.

**For MTR Trainning**

```bash
# make sure the working directory = `llm_augmented_mtr/tools`
source run/dist_train.sh
```

**For MTR Evaluation**

```bash
# make sure the working directory = `llm_augmented_mtr/tools`
source run/dist_valid.sh
```

**For MTR Testing**

```bash
# make sure the working directory = `llm_augmented_mtr/tools`
source run/dist_train.sh
```

## 5.2 I want to train this model on 20% WOMD

We currently only provide llm-augmented module on 100% WOMD, if you want to train llm-augmented-mtr on 20% WOMD, there are several works you need to do.

**Why?** You indeed can train llm-augmented-mtr on 20% WOMD by change the `--cfg_file` in scripts in `tools/run` folder from `cfgs/waymo/mtr+100_percent_data_llm_augmented.yaml` to `cfgs/waymo/mtr+20_percent_data_llm_augmented.yaml`. However, in this case, we still used the encoder of MTR trained on 100% WOMD, as `ENCODER_FOR_CONTEXT` in the configuration file is `100` (that's why the name of you downloaded context data includes `encoder_100`). So, we need to re-generate the `context_data_encoder_20.pkl` for WOMD.

**Make sure your working directory is `llm_augmented_mtr/LLM_integrate`**

**Step 0.** Make sure you have trained MTR on 20% WOMD, and choose a ckpt that have the best performance.

You can know the best checkpoint from `best_eval_record.txt` in the training output folder, like `output/waymo/mtr+100_percent_data_llm_augmented/llm_augmented_mtr+100_percent/eval/eval_with_train`.

**Step 1.** Convert the raw data provided by LLM to formatted file

We use LLM to generate context data for 14,000 agents in WOMD validation set (in the folder `LLM_output/raw_data`). At first, we need to convert its format so that the dataloader can load corresponding data in the WOMD.

```bash
# in `LLM_output/format_convert.py` change the validation_set_path to your WOMD path
python LLM_output/format_convert.py
```

After excute the python script, two file will generate:

- `LLM_output/info_file/llm_used_dataset_info.py`: the info file for dataloader.
- `LLM_output/context_file/llm_output_context.py`: context file for our later retrieval usage.

**Step 2.** Generate embeddings (feature vectors) for those data with context information

We have encapsulate the process for you!

At first, change some parameters:

1. in `LLM_integrate/tools/cfgs/generate_feature_vector.yaml`, change `LLM_OUTPUT_CONTEXT_PATH` to the path of generated file in **Step 1**
2. in `LLM_integrate/tools/scripts/run_embedding_generate.sh`, change the `--ckpt`,  according to your ckpt path and `--extra-tag` to `encoder_20`.

Next, excute this script.

```bash
# working directory is `llm_augmented_mtr/LLM_integrate`
cd tools
source run_embedding_generate.sh
```

Then, your generated context data in **Step 1** will equipped with corresponding embedding via MTR encoder.

The file will stored in the folder `LLM_output/context_file/llm_output_context_with_encoder_20.pkl`

**Step 3.** Do KNN retrieval to generate context data for the whole WOMD

There are so many data in WOMD, and we only generate context data for a minor part of WOMD. So we choose generate context data for the whole WOMD based on this minor part (like semi-supervise learning).

We use generated embedding to calculate euclidean distance to do KNN retrieval for each data in the WOMD.

You can do retrieval by our encapsulate script. What you need to do is change following parameters in `scripts/run_generate_context_for_whole_dataset.sh`:

1. `--ckpt`
2. `extra_tag`: to `encoder_20`
3. `retrieval_database`: to the path of file that generated in **Step 2**

```bash
# working directory is `llm_augmented_mtr/LLM_integrate`
cd tools
source run_generate_context_for_whole_dataset.sh
```

Finally, you will get context data for the whole dataset like you downloaded.

**Step 4.** Change configureation file and run

At the last step, just change `--cfg_file` from `cfgs/waymo/mtr+100_percent_data_llm_augmented.yaml` to `cfgs/waymo/mtr+20_percent_data_llm_augmented.yaml`, as well as `--ckpt` and `--extra_tag`.

```bash
# working directory is `llm_augmented_mtr`
cd tools
source dist_train_llm_augmented_mtr.sh
```

# 6. Other Questions

If you have other questions regarding this repo, please create an issue, we will give feedback☺.