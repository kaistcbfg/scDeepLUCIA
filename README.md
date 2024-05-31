# scDeepLUCIA
identify chromatin loops from single cell HiCAR dataset.

## Input data
- One-hot encoded genomic feature npy file.
- 25bp-resolution epigenomic feature npy file.
- 5kb reesolution contact matrix npy file.

## Parameters
- cut-off for the genomic/epigenomic score.
- cut-off for the contatct matrix-derived score.
- min & max length of the chromatin loops to find.
- loop clustering distance threshold.

## Output files
- identified chromatin loops in bedpe format

## Dependencies
- tensorflow==2.6.0
- pandas==1.1.3
- scikit-learn==0.22

## Example 
- [jupyter notebook for the astrocyte chr10](scDeepLUCIA_demo.ipynb)


