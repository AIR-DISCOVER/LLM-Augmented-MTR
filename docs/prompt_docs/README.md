before run our project, you need first clone it to your local server.

```bash
git clone git@github.com:SEU-zxj/LLM-Augmented-MTR.git
```

# 1. Install

Install all the required packages

```bash
cd llm_augmented_prompt
pip install -r requirements.txt
```

# 2. Generate TC-Map

Before generate TC-Map, first make sure you have got **validation dataset** of [Waymo Open Motion Dataset (WOMD)](https://waymo.com/open/download/), you can refer to [Dataset Preparation](https://github.com/SEU-zxj/LLM-Augmented-MTR/blob/main/docs/model_docs/README.md#1-dataset-preparation).

Run file of TC-Map generation.

```bash
cd tc_map
python generate.py
```

# 3. Get Context from GPT4-V

Run file of context generation.

```bash
cd prompt
python result_quantify.py
```

Moreover, you can use `--bt` to define batch_tag, use `--bz` to define batch_size, use `--key` to define using which key in key_list, use `--on` to define output_name of output folder.

```bash
python result_quantify.py --bt 1 --bz 500 --key 1 --on test
```

# 4. Other Questions

If you have other questions regarding this repo, please create an issue, we will give feedback☺.


