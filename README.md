# Orthogonal Gated Recurrent Unit with Neumann-Cayley Transformation

+ Code is going to be available soon)
+ Requered packages: Tensorflow/PyTorch, NumPy

## PARENTHESIS TASK
```
cd parenthesis_task
```

### Requirements
#### Requirements
##### Requirements
###### Requirements
```
pip install -r requirements.txt
```

## Training Parenthesis Task from scratch
To train LSTM model: 
```
bash run_parenthesis_LSTM.sh
```
To train GRU model: 
```
bash run_parenthesis_GRU.sh
```
To train scoRNN model: 
```
bash run_parenthesis_scoRNN.sh
```
To train GORU model: 
```
bash run_parenthesis_GORU.sh
```
To train NC-GRU(U_c) model (NC-GRU method only inside the U_c weight): 
```
bash run_parenthesis_NCGRU_1.sh
```
To train NC-GRU(U_c,U_r) model (NC-GRU method inside both the U_c and U_r weights): 
```
bash run_parenthesis_NCGRU_2.sh
```


# DENOISE TASK
```
cd denoise_task
```

## Requirements
```
pip install -r requirements.txt
```

## Training Denoise Task from scratch
To train LSTM model: 
```
bash run_denoise_LSTM.sh
```
To train GRU model: 
```
bash run_denoise_GRU.sh
```
To train scoRNN model: 
```
bash run_denoise_scoRNN.sh
```
To train GORU model: 
```
bash run_denoise_GORU.sh
```
To train NC-GRU(U_c) model: 
```
bash run_denoise_NCGRU_1.sh
```
To train NC-GRU(U_c,U_r) model: 
```
bash run_denoise_NCGRU_2.sh
```


# CHARACTER-LEVEL PENN TREEBANK
```
cd character_Penn_TreeBank
```

## Requirements
```
pip install -r requirements.txt
```

## Training Denoise Task from scratch
To train NC-GRU(U_c) model: 
```
bash run.sh
```
Note: You should be able to obtain bpc of ~1.385 with settings provided in run.sh file.

If you use this code or our results in your research, please cite as appropriate:

```
@misc{https://doi.org/10.48550/arxiv.2208.06496,
  doi = {10.48550/ARXIV.2208.06496},
  url = {https://arxiv.org/abs/2208.06496},
  author = {Mucllari, Edison and Zadorozhnyy, Vasily and Pospisil, Cole and Nguyen, Duc and Ye, Qiang},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Orthogonal Gated Recurrent Unit with Neumann-Cayley Transformation},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```


## Acknowledgments
Project is built in parts on top of 
- [AWD-LSTM](https://github.com/salesforce/awd-lstm-lm),
- [scoRNN](https://github.com/SpartinStuff/scoRNN),
- [GORU](https://github.com/jingli9111/GORU-tensorflow), and
- [ISAN](https://github.com/philipperemy/tensorflow-isan-rnn)
