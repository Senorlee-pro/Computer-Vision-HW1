# Computer-Vision-HW1
## Environment Setup
You may want to create a new `conda` environment for this project.

```bash
conda create -n myenv python=3.10 -y
conda activate myenv
```

Then, enter the project directory and install the required dependencies:

```bash
pip install -r requirements.txt
```
## Dataset
You should put the FashionMNIST dataset folder in `data` folder in the root directory, such as `root/data/FashionMNIST`.

```bash
root/
├── data/
│   └── FashionMNIST
├── dataset
├── models
├── requirements.txt
├── test.py
├── train.py
└── visualize.py
```

## Project Structure
- **`dataset/`** - Includes the implementation of dataloader and datasubset.
- **`models/`** - Includes the definition and implementation of fundamental blocks, MLPClassifier and the loss function.
- **`test.py`** - Implements the testing of best weights on test set, plotting the confusion matrix and visualizing misclassified images.
- **`train.py`** - Implements the main training process `run`, grid search for hyperparameters and other tool functions for training process visualization.
- **`visualize.py`** - Implements the visualization for the weights of the first FC layer.

## Get started
For training process, you should use `run` function in `train.py`, whose key parameters are listed below:
`hidden_dims`: a three-dimensional tuple, corresponding to the dimensions of three hidden layers
`epochs`: number of training epochs
`lr`: learning rate
`weight_decay`: weight decay rate, normalization coefficient
`batch_size`: batch size, defaults to 128
`lr_decay`: learning rate decay, defaults to 0.9
`activation_strategy`: the activation function you wish to use in non-linear layers, including `relu`, `sigmoid`, `tanh`, defaults to `relu`
`save_weights`: whether to save the best model weights to local directory, named `best_model.npz`, defaults to `False`
