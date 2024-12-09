# SAD-VER

Code for research paper:

SAD-VER: A Self-supervised, Diffusion probabilistic model-based data augmentation framework for Visual-stimulus EEG Recognition (Submitted to _Advanced Engineering Informatics_, in revision).

<img width="2864" alt="fig1_overall_framework" src="https://github.com/user-attachments/assets/b1eb458b-766f-4902-8135-e79730969a49">



__Detailed Introduction:__

_0. Requirements_

Environments: Python >= 3.7 / PyTorch >= 1.10 / CUDA >= 11.3 / Ubuntu 20.04 or Windows 11 23H2.

The number of trainable parameters in the U-Net used by AV-DPM is approximately 146.21M, with a GPU memory usage of about 6GB. To deploy SAD-VER, a NVIDIA GPU with VRAM >= 8GB is recommended.

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

We also validate SAD-VER's performance on SEED & SEED-IV dataset. Codes avaliable at _SADVER_Series_E_ (E stands for Emotional EEG). To use the SEED and SEED-IV datasets, you first need to download the SEED and SEED-IV datasets and run the mapping.py file in the _Data Preprocessing_ folder. Then, replace the corresponding files in AV-DPM and STI-Net with the files from _SADVER_Series_E_.

SEED and SEED-IV datasets are avaliable at: https://bcmi.sjtu.edu.cn/home/seed/

Our experimental results in SEED & SEED-IV dataset are included in our revision to reviewers. They will be released in public soon.

_5. Acknowledgements_

This project is deeply inspired by @ermongroup https://github.com/ermongroup/ddim . Salute to all open-source researchers!

Produced by Laboratory of Brain-Inspired Intelligence & Human-Computer Interaction, Jilin University, China.

Supervised by Prof. Wanzhong Chen & Prof. Mingyang Li.
