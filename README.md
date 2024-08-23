# SSVD

# Less is More: Unlocking Semi-Supervised Deep Learning for Vulnerability Detection

## Introduction

This is the official implementation of **S**emi-**S**upervised **V**ulnerability **D**etection (SSVD). SSVD is a semi-supervised learning method for software vulnerability detection. In this project, we provide the **code implementation** of the relevant paper, as well as **all the datasets and baseline codes** mentioned in the paper. We welcome all researchers interested in our work to reproduce the results. You can download the data needed for this project [here](https://figshare.com/s/e4c4cdf09b800bfc3643).

## Environment Setup

```
pip intsall numpy
pip install torch # We used torch 2.0.1 in our paper.
pip install transformers
pip install sklearn
pip install scikit-learn
pip install balanced-loss
pip install pandas
pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install torch_geometric
```

## Dataset and CodeBert

### Dataset

We utilize the cleaned versions of **Devign**, **Big-Vul**, and **Juliet** provided by Roland et al. [Here](https://figshare.com/articles/software/Reproduction_Package_for_Data_Quality_for_Software_Vulnerability_Datasets_/20499924) is the cleaned version data link. We also used the original **ReVea**l dataset as it does not have a clean version. To adapt to both LineVul and ReVeal basic detection models, we processed the four datasets into **text datasets** and **graph datasets** separately. 

1. To facilitate the reproduction of our work, we have directly placed the text dataset in `/LineVul_SSVD/ssl_data/`. After extracting the `.zip` file, you will obtain `.pkl` files, each `.pkl` file represents a code snippet.

2. For graph datasets,  Many works utilize Joern to extract graphs from source code. However, due to the difficulty in configuring the Joern environment, we provide the preprocessed graph dataset using Joern in `/ReVeal_SSVD/ssl_data/`. After extracting the `.zip` file, you will obtain `.pkl` files, each `.pkl` file represents a code snippet.
3. To perform five-fold cross-validation and explore the impact of different labeled data proportions, the dataset needs to be divided. The `slice.py` script can divide the dataset into 11 different labeled data proportions and five cross-validation datasets:

```
cd {model}_SSVD/ssl_data/
python slice.py --dataset {DATASET_NAME}
```

### CodeBert

[Here](https://huggingface.co/microsoft/codebert-base) is the official CodeBert. For the convenience of reproduction, we have downloaded and placed it in the `/LineVul_SSVD/codebert/` directory.

## Run SSVD

### Get Teacher Model

First, we need to initialize the teacher model.

```
python python build_teacher.py --dataset {DATASET_NAME} --label_rate {LABEL_RATIO} --base_model {BASE_MODEL} --stopper_mode {STOPPER_MODE} --cross_val
```

`--dataset`: Choose the dataset on which to train. `Devign`, `reveal`, `Juliet`, or `big-vul`.

`--label_rate`: Percentage of labeled data. `0.1`, `0.2`, ..., `1.0`. `1.0` represents full supervision.

`--base_model`: Base detection model. `LineVul` or `ReVeal`.

`--stopper_mode`: Stop training when there is no improvement in the validation set's F1/accuracy. `f1` or `acc`.

`--cross_val`: If you need to perform cross-validation, add this parameter.

After training is completed, the teacher model will be stored in the path `{BASE_MODEL}_SSVD/teachers/{DATASET_NAME}/{slice_i}/{LABEL_RATIO}/{STOPPER_MODE}/`. Please copy the `best_teacher.pth` from there to the `{BASE_MODEL}_SSVD/teachers/{DATASET_NAME}/{slice_i}/{LABEL_RATIO}`.

### Get Student Model

Second, after obtaining the teacher model, we can use this model to generate pseudo-labels and start the teacher-student iteration.

```
python python build_student.py --dataset {DATASET_NAME} --label_rate {LABEL_RATIO} --base_model {BASE_MODEL}  --cross_val
```

The meanings and available options for the parameters are the same as in `build_teacher.py`.

- We store the settings for SSVD in `{BASE_MODEL}_SSVD/config.yaml`. If needed, you can modify the parameter configurations in that file.
- We store the SSVD iteration results in `{BASE_MODEL}_SSVD/checkpoint/{slice_i}/{DATASET_NAME}/{LABEL_RATIO}/`.
- From the model weight file, you can directly observe the improvement of the student model relative to the teacher model. For example, `best_student_0.00120_0.14000.pth` represents that the student model has improved in accuracy by 0.00120 and in F1-score by 0.14000 compared to the teacher model in the test dataset.
- For the convenience of result analysis, we also automatically save the results in Excel format and place them in the directory `{BASE_MODEL}_SSVD/excels/checkpoint/{slice_i}/{DATASET_NAME}/{LABEL_RATIO}/`. The Excel file includes metrics for the teacher model, metrics for the student model, and metrics for the pseudo-labels.

## Baselines

We directly implemented partial baselines (SST, UST, HADES, DST) within the SSVD framework. Similarly, we re-implemented some baselines (co-training, tri-training, LP) and also utilized open-source code for other baselines (PILOT). Here are the corresponding script files and project links for the baselines:

```
# For HADES
python baseline_HADES.py --dataset {DATASET_NAME} --label_rate {LABEL_RATIO} --base_model {BASE_MODEL} --cross_val
# For SST
python baseline_standardST.py --dataset {DATASET_NAME} --label_rate {LABEL_RATIO} --base_model {BASE_MODEL} --cross_val
# For SSVD(prob)
python baseline_conf_threshold.py --dataset {DATASET_NAME} --label_rate {LABEL_RATIO} --base_model {BASE_MODEL} --cross_val
# For DST
python baseline_DST.py --dataset {DATASET_NAME} --label_rate {LABEL_RATIO} --base_model {BASE_MODEL} --cross_val
```

co-training, tri-training, and LP are in the  `baseline.zip`. Since running these methods requires the `cls` vectors output by CodeBert, we have also provided the `cls` vector dataset in the `baseline.zip` file.

DST: [Link](https://github.com/thuml/Debiased-Self-Training)

PILOT: [Link](https://github.com/PILOT-VD-2023/PILOT)

For the convenience of reproduction, we have also placed the project file for PILOT in the `PILOT.zip` file.

- For PILOT, we implemented it directly using the official code. The only difference is that the official PILOT code uses a dataset format that is different from ours. Therefore, we converted the dataset format before running PILOT. We placed the converted dataset under the directory `dataset/`. The code for dataset format conversion can be found in `dataset/SSVD_data/SSVD2PILOT.py`. You can run `.sh` files to run PILOT. The execution sequence is as follows: First, run `train_1.sh`, then run `step_1.sh`, followed by running `train_2.sh`, and finally run `train_iterative.sh`.

### References


> [1] Xin-Cheng Wen, Xinchen Wang, Cuiyun Gao, Shaohua Wang, Yang Liu, and Zhaoquan Gu. 2023. When Less is
> Enough: Positive and Unlabeled Learning Model for Vulnerability Detection. arXiv preprint arXiv:2308.10523 (2023)
>
> [2] Subhabrata Mukherjee and Ahmed Awadallah. 2020. Uncertainty-aware self-training for few-shot text classification.
> Advances in Neural Information Processing Systems 33 (2020), 21199–212
>
> [3] Baixu Chen, Junguang Jiang, Ximei Wang, Pengfei Wan, Jianmin Wang, and Mingsheng Long. 2022. Debiased
> self-training for semi-supervised learning. Advances in Neural Information Processing Systems 35 (2022), 32424–32
>
> [4] Dong-Hyun Lee et al . 2013. Pseudo-label: The simple and efficient semi-supervised learning method for deep neural
> networks. In Workshop on challenges in representation learning, ICML, Vol. 3. Atlanta, 89