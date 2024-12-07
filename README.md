# SAD-VER

Code for research paper:

SAD-VER: A Self-supervised, Diffusion probabilistic model-based data augmentation framework for Visual-stimulus EEG Recognition (Under Journal's Consideration).

<img width="2864" alt="fig1_overall_framework" src="https://github.com/user-attachments/assets/b1eb458b-766f-4902-8135-e79730969a49">



__Detailed Introduction:__

_1. Data Preprocessing_

Dataset (Public EEG dataset from Stanford Digital Repository, OCED) utilizd in this research is avaliable at: https://purl.stanford.edu/bq914sc3730

Run mapping.py to obtain EEG data mapped in a 13x13 2D grid. The EEG data shape should be transformed from (124, 32, samp_num) to (samp_num, 32, 13, 13).

_2. AV-DPM_

We have made the complete training and sampling process of AV-DPM public. The relevant settings can be adjusted according to your needs in main.py and OCED.yml file.

_3. STI-Net_

STI-Net folder contains all the decoding networks utilized in our research (STI-Net, EEGNet, EEGConformer, etc.). The complete training & validating pipeline is included.

Note that before using these networks, you should specify the location where the dataset is stored by yourself, 
and organize it according to whether it is for generating EEG, or whether it uses 124 channels or 13x13 mapping. The default storage location for the dataset is ./EEG_data.

We have provided a preprocessed dataset via Baidu Netdisk. It can be easily accessed in mainland China, but this cannot be guaranteed in other regions. 

You can access them on https://pan.baidu.com/s/1u1JRxspI6VCk9Q-788LfAA , password: oril . 
Just place it under the STI-Net folder and it should work well.

_4. Validation on Other Datasets_

We also validate SAD-VER's performance on SEED & SEED-IV dataset. Codes will be released before Dec. 10th.

_5. Acknowledgements_

Produced by Laboratory of Brain-Inspired Intelligence & Human-Computer Interaction, Jilin University, China.

Supervised by Prof. Wanzhong Chen & Prof. Mingyang Li.
