# Context-Modeling-with-Multimodal-Prompts-for-Emotion-Recognition-in-Conversation

### Requirements
+ python 3.9.0
+ torch 1.10.1
+ CUDA 11.3

### Dataset

The pre-extracted features and the checkpoints are avaliable at [here](https://pan.baidu.com/s/1CS_UyqSRrwG2FQ0W1Vz1Mw?pwd=rtwp). code: rwtp

### Test
Test on MELD and IEMOCAP using the checkpoints:
```
python -u test.py --base-model 'GRU'  --batch-size 16 --epochs=0 --multi_modal --fusion='TF' --modals='avl' --Dataset='dataset' --norm BN
```
