# SAD-VER

Code for research paper:

SAD-VER: A Self-supervised, Diffusion probabilistic model-based data augmentation framework for Visual-stimulus EEG Recognition (Submitted to _Advanced Engineering Informatics_, in revision).

<img width="2864" alt="fig1_overall_framework" src="https://github.com/user-attachments/assets/b1eb458b-766f-4902-8135-e79730969a49">



__Detailed Introduction:__

**0. Requirements**

Environments: Python >= 3.7 / PyTorch >= 1.10 / CUDA >= 11.3 / Ubuntu 20.04 or Windows 11 23H2.

The number of trainable parameters in the U-Net used by AV-DPM is approximately 146.21M, with a GPU memory usage of about 6GB. To deploy SAD-VER, a NVIDIA GPU with VRAM >= 8GB is recommended.

**1. Data Preprocessing**

Dataset (Public EEG dataset from Stanford Digital Repository, OCED) utilizd in this research is avaliable at: https://purl.stanford.edu/bq914sc3730

Run _mapping.py_ to obtain EEG data mapped in a 13x13 2D grid. The EEG data shape should be transformed from (124, 32, samp_num) to (samp_num, 32, 13, 13).

**2. AV-DPM**

We have made the complete training and sampling process of AV-DPM public. The relevant settings can be adjusted according to your needs in _main.py_ and _OCED.yml_ file.

**3. STI-Net**

STI-Net folder contains all the decoding networks utilized in our research (STI-Net, EEGNet, EEGConformer, etc.). The complete training & validating pipeline is included.

Note that before using these networks, you should specify the location where the dataset is stored by yourself, 
and organize it according to whether it is for generating EEG, or whether it uses 124 channels or 13x13 mapping. The default storage location for the dataset is ./EEG_data.

We have provided a preprocessed dataset via Baidu Netdisk. It can be easily accessed in mainland China, but this cannot be guaranteed in other regions. 

You can access them on https://pan.baidu.com/s/1u1JRxspI6VCk9Q-788LfAA , password: oril . 
Just place it under the STI-Net folder and it should work well.

**4. Validation on Other Datasets**

We also validate SAD-VER's performance on SEED & SEED-IV dataset. Codes avaliable at _SADVER_Series_E_ (E stands for Emotional EEG). To use the SEED and SEED-IV datasets, you first need to download the SEED and SEED-IV datasets and run the mapping.py file in the _Data Preprocessing_ folder. Then, replace the corresponding files in AV-DPM and STI-Net with the files from _SADVER_Series_E_.

SEED and SEED-IV datasets are avaliable at: https://bcmi.sjtu.edu.cn/home/seed/

Our experimental results in SEED & SEED-IV dataset are included in _Results on SEED.pdf_.

**5. Acknowledgements**

This project is deeply inspired by @ermongroup https://github.com/ermongroup/ddim . Salute to all open-source researchers!

Produced by Laboratory of Brain-Inspired Intelligence & Human-Computer Interaction, Jilin University, China.

Supervised by Prof. Wanzhong Chen & Prof. Mingyang Li.
