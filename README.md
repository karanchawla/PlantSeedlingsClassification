This repository contains my models for Kaggle competitions. 

1. [Plant Seedling Competition](https://www.kaggle.com/c/plant-seedlings-classification) - [Code](https://github.com/karanchawla/kaggle-dabble/blob/master/competitions/plantseedling.md)

Key Ideas used: 
1. Transfer Learning on ResNet101 architecture 
2. Trained the penultimate layer first 
3. Cyclic Learning Rate for finding the best minimum

![png](https://github.com/karanchawla/kaggle-dabble/blob/master/Images/output_17_0.png)

4. Used [Ref](https://arxiv.org/pdf/1506.01186v6.pdf) for finding the best learning rate

![png](https://github.com/karanchawla/kaggle-dabble/blob/master/Images/output_13_0.png)

5. Retrained the complete architecture with lower learning rates like so 
```
lrs = np.array([lr/9,lr/3,lr])
learn.unfreeze()
learn.fit(lrs, 3, cycle_len=2, cycle_mult=2)
```
6. Used test time tranformations for higher accuracy predictions. 
