# Time series UDA

# TODO:
* Add features (paper: Multi-Scale Convolutional Neural Networks for Time Series)
* Data augmentation?
* Hyperparameters tuning
* Data better preprocessing (amplitudes, ARMA)

Here we applied paper [Asymmetric Tri-training for Unsupervised Domain Adaptation](https://arxiv.org/abs/1702.08400)
to time series [Sussex-Huawei locomotion (SHL)](http://www.shl-dataset.org/) dataset. Idea was to implement identical
training pipeline as in original paper (image recognition UDA) with modified feature extraction model. We adopted 
feature extracting techniques from 
[Benchmarking the SHL Recognition Challenge with Classical and Deep-Learning Pipelines](http://acm.mementodepot.org/pubs/proceedings/acmconferences_3267305/3267305/3267305.3267531/3267305.3267531.pdf).

However, we got results not as well as in CV UDA:

|Before UDA (pretrained on source) | After UDA |
|---|---|
|29.3% | 36.1% (Naive CNN feature extractor)|
|29.0% | 33.3% (extractor from [paper](http://acm.mementodepot.org/pubs/proceedings/acmconferences_3267305/3267305/3267305.3267531/3267305.3267531.pdf))|

The quality of models trained on source are trustful, because they achieved accuracy around 84.7% - result close to
[paper's](http://acm.mementodepot.org/pubs/proceedings/acmconferences_3267305/3267305/3267305.3267531/3267305.3267531.pdf) results when used with CNN architecture (86.6% mentioned). 



## raw SHL preprocessing

Download from http://www.shl-dataset.org/download/

```Version 1 as zip: Part 1 (2.7GB), Part 2 (2.3GB), Part 3 (2.1GB)``` as *target* domain data

```SHLDataset_User1Hips_v1.zip.001; SHLDataset_User1Hips_v1.zip.002; SHLDataset_User1Hips_v1.zip.003; SHLDataset_User1Hips_v1.zip.004; SHLDataset_User1Hips_v1.zip.005```
as *source* domain data

1. place data like
    ```
    ├── data
    │  └── raw
    │      ├── 220617
    │      │  ├── Hand_Motion.txt
    │      │  ├── Hips_Motion.txt
    │      │  ├── Label.txt
    │      │  └── Torso_Motion.txt
    │      ├── 260617
    │      │   ├── Hand_Motion.txt
    │      │   ├── Hips_Motion.txt
    │      │   ├── Label.txt
    │      │   └── Torso_Motion.txt
    │      ├── ...
    ```

2. run ``` shl_processing.py ``` with relevant paths

3. check with
    ```
        python dataloader.py
    ```
   
## Training
```
python sovler.py
```
