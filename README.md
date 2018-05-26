# VAEGAN
Implementation of [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300v2) in Keras.

## Prerequisites

- tensorflow >=1.4
- keras >= 2.1.4
- OpenCV >= 3.4.0
- numpy

## Dataset 

[CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 

## Usage
  
   Train:
    
    $ python main.py --op 0 --path your data path
  
  Test:
  
    $ python main.py --op 1 --path your data path

## Image Generation from Noise Results

 
    
[//]: <> (![](img/real.png))
    
 
    
[//]: <> (![](img/recon.png))
    
    
Problems Encountered

-Hard to train.
-Numerical instability in loss.


