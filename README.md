# SiamFC-Pytorch
Full convolutional Siamese network implementation using Pytorch with ```GOT10k``` training set.

## Architecture
The SiamFC algorithm suggests learning a function f(z,x) that compares a template image z with a candidate image x of the same size and returns a high score if the two images describe the same object, otherwise a low score.

## Object Tracking

```bash
cd SiamFC_Pytorch
python run_tracker.py --data_dir path/to/data --model_dir path/to/model
```

## Training

Download GOT-10K

```bash
cd SiamFC_Pytorch
# data preprocessing
python data_preprocessing.py --data-dir path/to/data/GOT-10K \
			     --output-dir path/to/data/GOT-10K/crop_data \
			     --num_processings 8
# training 
python train.py --train_data_dir path/to/data/GOT-10K/crop_train_data  \
			     --val_data_dir path/to/data/GOT-10K/crop_val_data 
```

#### Reference

[NeverGX/siamfc-pytorch](https://github.com/NeverGX/siamfc-pytorch)
