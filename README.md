# scDeepLUCIA
identify chromatin loops from single cell HiCAR dataset.

## Input data
- One-hot encoded genomic feature npy file.
- 25bp-resolution epigenomic feature npy file.
- 5kb-resolution contact matrix npy file.

## Parameters
- cut-off for the genomic/epigenomic score.
- cut-off for the contatct matrix-derived score.
- min & max length of the chromatin loops to find.
- loop clustering distance threshold.

## Output files
- identified chromatin loops in bedpe format

## Dependencies
- tensorflow>=2.14.0 ( tested with 2.14.0 )
- pandas>=2.1.4 ( tested with 2.1.4 ) 
- scikit-learn>=1.4.2 ( tested with 1.4.2 )

## Container
- Tested with Amazon sagemaker distribution 1.8.0-gpu.

## Hardware recommendation 
- NVIDIA GPU with Compute Capability >= 7.5

## Installation guide
```
git clone https://github.com/kaistcbfg/scDeepLUCIA.git
```

## Example 
- [jupyter notebook for the astrocyte chr10](scDeepLUCIA.ipynb)


