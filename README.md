# SAD-VER

Code for research paper:

SAD-VER: A Self-supervised, Diffusion probabilistic model-based data augmentation framework for Visual-stimulus EEG Recognition (Under Journal's Consideration).

<img width="2864" alt="fig1_overall_framework" src="https://github.com/user-attachments/assets/b1eb458b-766f-4902-8135-e79730969a49">



__Detailed Introduction:__

_1. Data Preprocessing_

Dataset (Public EEG dataset from Stanford Digital Repository, OCED) utilizd in this research is avaliable at: https://purl.stanford.edu/bq914sc3730

Run mapping.py to obtain EEG data mapped in a 13x13 2D grid. The EEG data shape should be transformed from (124, 32, samp_num) to (samp_num, 13, 13, 32).

_2. AV-DPM_

_3. STI-Net_

_4. Acknowledgements_

Produced by Laboratory of Brain-Inspired Intelligence & Human-Computer Interaction, Jilin University, China.

Supervised by Prof. Wanzhong Chen & Prof. Mingyang Li.
