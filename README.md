# SpectralNetAutoEncoder

## Context
This repository has been done during my last year as an engineer student working on a project on Spectral Embedding.  
It is mostly based on the SpectralNet github and rewrite some of their functions.  
Thanks to the SpectralNet formulation of Spectral Clustering using Neural Networks, it is possible to use different methods on the embedding.   
The goal of this repository is to use an AutoEncoder regularization on SpectralNet in order to keep as much information on the embedding as possible.  

## Installation
You can download dependencies using :  
```bash
pip install -r requirements.txt
``` 
You might need to manually download the munkres and annoy wheels.  

## Usage example
```python
from SpectralNetAE import SpectralNetAE
from utils import load_data
import torch

# Fetch the dataset
x_train, x_test, y_train, y_test = load_data("two_moons")

X = torch.cat([x_train, x_test])
y = torch.cat([y_train, y_test])

# Configure the neural network
spectralnetae = SpectralNetAE(
        n_clusters = 2,
        spectral_batch_size = 1024,
        spectral_max_epochs = 1000,
        spectral_tolerance  = 1e-9,
        spectral_is_local_scale = False, 
        spectral_num_neighbours = 8,
        spectral_scale_k = 2,
        spectral_lr = 1e-2,
        spectral_config = {"hiddens": [128, 128, 2]},
        spectral_is_normalized = False,
        spectral_alpha =  2e-3,
    )

# Train SpectralNetAE
spectralnetae.fit(X)
cluster_assignments = spectralnetae.predict(X) # Get the final assignments to clusters
```

## Results sample
Here, we can see that using AutoEncoder regularization, the spectral embedding is able to keep more information than the SpectralNet's.  

<p align="center">
    <img src="https://github.com/Arthedon/SpectralNetAutoEncoder/blob/main/figure/embedding_comparison.png">

With circles dataset on top and two moons dataset on bottom.  

Some performances has been tracked for different dataset using the same architecture for SpectralNet, SpectralNet with AutoEncoder and AutoEncoder only :
<table>
  <tr><td colspan="1"><center>Datasets</td><td colspan="3"><center>Circles</td><td colspan="3"><center>Two Moons</td><td colspan="3"><center>MNIST</td></tr>
  <tr><td>Networks</td><td>SpectralNet</td><td>SpectralNetAutoEncoder</td><td>AutoEncoder</td><td>SpectralNet</td><td>SpectralNetAutoEncoder</td><td>AutoEncoder</td><td>SpectralNet</td><td>SpectralNetAutoEncoder</td><td>AutoEncoder</td></tr>
  <tr><td><center>ACC</td><td><center>100%</td><td><center>100%</td><td><center>55.9%</td><td><center>100%</td><td><center>100%</td><td><center>87.3%</td><td><center>60.3%</td><td><center>75.1%</td><td><center>80.2%</td></tr>
  <tr><td><center>NMI</td><td><center>99.8%</td><td><center>100%</td><td><center>1%</td><td><center>100%</td><td><center>100%</td><td><center>45%</td><td><center>62.3%</td><td><center>73.5%</td><td><center>73.4%</td></tr>
  <tr><td><center>α</td><td><center>0</td><td><center>4,5.10^−5</td><td><center>-</td><td><center>0</td><td><center>2,10^-3</td><td><center>-</td><td><center>0</td><td><center>2,275.10^4</td><td><center>-</td></tr>
</table>
ACC : Accuracy; NMI : Normalized Mutual Information; α : Regularization term 

## References
From the paper https://arxiv.org/abs/1801.01587 by Shaham U., et al. and their GitHub https://github.com/shaham-lab/SpectralNet/tree/main