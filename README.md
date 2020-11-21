# Supplementary Material (PaperID: 5139)

Source code for reviewers.
Please refer to the followings for dataset settings, pre-trained models, and evaluation.

## Environment
- PyTorch 1.5.0
- opencv-python
- PIL (pillow)
- tqdm

## Datasets

1. Download databases
- [UCSD Ped2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html)
- [CUHK Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
- [Shanghai Tech Campus](https://svip-lab.github.io/dataset/campus_dataset.html)

2. Save the frame of each video clip as an image. Make 'database' directory and place data as follows.
```bash
database

├── Ped2

	├── training
  
		├── frames
    
	├── testing
  
		├── frames
    
├── Avenue

	├── ...
  
└── shanghaitech

	├── ...
```

## Train

Train code will be available after the final decision.

## Pre-trained Models

1. Download our pre-trained models.

   [[pre-trained models]](https://drive.google.com/drive/folders/19IbKloMhysMOlfq1Hns1VqbjiYt2A-mx?usp=sharing) (This repository link is anonymized.)

2. Place pre-trained models in 'save' directory.

## Evaluation

Run Evaluation.py to evaluate the performance of trained models with following commands.

The result of every clips of each database is in the 'save' directory as a txt file.

- UCSD Ped2
    ```bash
    python Evaluation.py --datapath '/home/myeongah/AnomalyDetection/database' --dataset Ped2 --flow_L 1 --checkpoint './save/FINAL_Ped2/Backbone.pth' --checkpoint_flow_static './save/FINAL_Ped2/Flow_static.pth' --checkpoint_flow_dynamic './save/FINAL_Ped2/Flow_dynamic.pth' --modelsave 'FINAL_Ped2'
    ```

- CUHK Avenue
    ```bash
    python Evaluation.py --datapath '/home/myeongah/AnomalyDetection/database' --dataset Avenue --checkpoint './save/FINAL_CUHK/Backbone.pth' --checkpoint_flow_static './save/FINAL_CUHK/Flow_static.pth' --checkpoint_flow_dynamic './save/FINAL_CUHK/Flow_dynamic.pth' --modelsave 'FINAL_CUHK'
    ```

- Shanghai Tech Campus
    ```bash
    python Evaluation.py --datapath '/home/myeongah/AnomalyDetection/database' --dataset shanghaitech --checkpoint './save/FINAL_ST/Backbone.pth' --checkpoint_flow_static './save/FINAL_ST/Flow_static.pth' --checkpoint_flow_dynamic './save/FINAL_ST/Flow_dynamic.pth' --modelsave 'FINAL_ST'
    ```

