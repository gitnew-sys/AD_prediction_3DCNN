# Volumetric CNN for Alzheimer's Disease Classification — PyTorch Implementation

Re-implementation of the method in:

> Oh, K., Chung, Y.-C., Kim, K. W., Kim, W.-S., & Oh, I.-S. (2019). *Classification
> and Visualization of Alzheimer's Disease using Volumetric Convolutional Neural
> Network and Transfer Learning.* Scientific Reports, 9, 18150.
> https://doi.org/10.1038/s41598-019-54548-6

## Pipeline (mirrors Fig. 2 of the paper)

```
Stage 1                Stage 2                     Stage 3
--------                --------                    --------
CAE/ICAE          →     AD vs. NC classifier   →     pMCI vs. sMCI classifier
(unsupervised            (encoder init'd              (encoder init'd from the
 reconstruction           from Stage-1 CAE,             *fine-tuned* AD/NC
 on AD+NC scans)          fine-tuned end-to-end)        encoder — transfer
                                                         learning — then
                                                         fine-tuned end-to-end)
                                                              ↓
                                                    saliency.py → biomarker maps
```

## Quick start (synthetic data, no ADNI access needed)

```bash
pip install -r requirements.txt
D:\AD_project\DCM2nii\convert_adni.ps1     <- Convert DICOM to NIfTI (one-time, before training)
python 3s_cnn.py                           <- the whole pipeline: data loading, model, training, saliency
```

## pMCI vs. sMCI labeling

The imaging data alone doesn't indicate whether an MCI subject converted to AD. That requires cross-referencing each subject's RID/PTID against ADNI's Diagnostic Summary table (DXSUM_PDXCONV_ADNIALL.csv, downloaded from LONI IDA), checking whether DIAGNOSIS changed to 3 (AD) within a 36-month follow-up window of the baseline MCI diagnosis (DIAGNOSIS == 2). Subjects that convert are pMCI; subjects that remain MCI/stable through 36 months are sMCI. This classification is done once, upstream, to sort scans into the data\pMCI / data\sMCI folders before running run_adni.py.

