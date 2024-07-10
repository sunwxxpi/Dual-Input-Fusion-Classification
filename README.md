# Dual Input Fusion Classification

- Train and Inference
```shell
CUDA_VISIBLE_DEVICES='0, 1' python train.py --data_path ./data/KO_ver_1/KO/train --class_num 5 --model_name custom --writer_comment KO_ver_1/KO/b_se --mode b_se && CUDA_VISIBLE_DEVICES='0, 1' python test.py --data_path ./data/KO_ver_1/KO/test --class_num 5 --model_name custom --writer_comment KO_ver_1/KO/b_se --mode b_se
```