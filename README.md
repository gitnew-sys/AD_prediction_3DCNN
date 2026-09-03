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

The imaging data alone doesn't indicate whether an MCI subject converted to AD. That requires cross-referencing each subject's RID/PTID against ADNI's Diagnostic Summary table (DXSUM_PDXCONV_ADNIALL.csv, downloaded separately from LONI IDA under Download > Study Data > Assessments > Diagnosis), checking whether DIAGNOSIS changed to 3 (AD) within a 36-month follow-up window of the baseline MCI diagnosis (DIAGNOSIS == 2). Subjects that convert are pMCI; subjects that remain MCI/stable through 36 months are sMCI. This classification is done once, upstream, to sort scans into the data\pMCI / data\sMCI folders before running run_adni.py.

## Honest caveats vs. the original paper

- Exact layer padding/stride choices are not fully specified in the text; this
  implementation uses "same" padding + 2×2×2 pooling, which reproduces the
  qualitative architecture but not necessarily the exact reported parameter
  counts (paper: ~1.44M for CAE, ~0.34M for ICAE).
- Nested 5-fold cross-validation (repeated 20×) and hyperparameter grid search
  over L1/L2 weighting (Fig. 5/8 in the paper) are described but not scripted
  here — `train.py` runs a single train/val split for clarity; wrap it in your
  own CV loop for a faithful reproduction of the evaluation protocol.
- DARTEL spatial normalization (SPM12) is external to this codebase.
- This code has been syntax-checked but **not executed against real MRI data
  or GPU hardware** in this environment (no network access to install PyTorch
  here) — please validate on your own machine before relying on results.
