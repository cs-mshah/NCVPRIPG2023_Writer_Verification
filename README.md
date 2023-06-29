# NCVPRIG 2023 Challenge on Writer Verification

Challenge Link: [Summer Challenge on Writer Verification, under NCVPRIPG'23](https://vl2g.github.io/challenges/wv2023)  

Kaggle Link: [Summer Challenge on Writer Verification](https://www.kaggle.com/competitions/writer-verification-on-summer-challenge/overview)  

## Environment

```shell
conda env create -f environment.yml
conda activate ncvp
```

## Evaluate CSV

```shell
python evaluate.py --label data/val.csv --pred outputs/MaSha_01.csv
```