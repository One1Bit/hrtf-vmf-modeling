# HRTF vMF Modeling
HRTF → SH → vMF

I start from HRTF data (SOFA format), represent it using spherical harmonics, 
and then model the spatial distribution with a mixture of von Mises–Fisher distributions.

Currently I am using a single subject from HUTUBS (`pp1_HRIRs_measured.sofa`) to test the approach. 

## Dependencies

- MATLAB
- [MixEst toolbox](https://github.com/utvisionlab/mixest)
- [HUTUBS dataset](https://api-depositonce.tu-berlin.de/server/api/core/bitstreams/8f6e24a2-1c75-4f84-a50f-74a34bb480c7/content)
