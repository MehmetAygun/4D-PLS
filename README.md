# Tracking evaluation for SemanticKITTI

This repo contains code for the paper 4D Panoptic Lidar Segmentation. 
The code is based on the Pytoch implementation of  <a href="https://github.com/HuguesTHOMAS/KPConv-PyTorch">KPConv</a>.


### Installation

```bash
git clone https://github.com/MehmetAygun/4D-PLS
cd 4D-PLS
pip install -r requirements.txt
cd cpp_wrappers
sh compile_wrappers.sh
```

### Data
Create a directory `data` in main directory, download the SemanticKITTI to there with labels from  <a href="http://semantic-kitti.org/dataset.html#download/">here</a>

Then create additional labels for training using utils/create_center_label.py, for testing this step is not necessary.

```bash
python create_center_label.py
```

The folder structure should be as follows:

```bash
data/SemanticKITTI/
└── sequences/
    └── 08/
        └── labels
            ├── 000000.label
            ├── 000001.label
            ...
            └── 004070.label
```

### Models

For saving models or using pretrained models create a folder named `results` in main directory. 
You can download a pre-trained model from <a href="https://drive.google.com/file/d/164ykCTdxwX7Wd_DsDyUYva4s_pFSfpAB/view?usp=sharing">here</a> .

### Training

For training, you should modify the config parameters in `train_SemanticKitti.py`.
The most important thing that, to get a good performance train the model using `config.pre_train = True` firstly at least for 200 epochs, then train the model using `config.pre_train = False`. 

```bash
python train_SemanticKitti.py
```

This code will generate config file and save the pre-trained models in the results directory.

### Testing & Tracking

For testing, set the model directory the choosen_log in `test_models.py`, and modify the config parameters as you wish. Then run :

```bash
python test_models.py
```

This will generate semantic and instance predictions for small 4D volumes under the test/model_dir. 
To generate long tracks using small 4D volumes use `stitch_tracklets.py`

```bash
python stitch_tracklets.py --predictions test/model_dir --n_test_frames 4
```
This code will generate predictions in the format of SemanticKITTI under test/model_dir/stitch .
