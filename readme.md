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