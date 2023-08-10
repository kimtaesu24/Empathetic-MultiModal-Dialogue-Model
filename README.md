# Multimodal-ConvAI

## Repository Structure

The overall file structure of this repository is as follows:

```
Multimodal-ConvAI
    ├── README.md                       
    ├── requirments.txt
    ├── run_reward.py                       # starts training the model with specified hyperparameters
    ├── demo.py                             # demo
    └── src         
        ├── train_reward.py                 # implements a function for training the model with hyperparameters
        ├── inference_reward.py             # implements a function for inference the model
        ├── utils
        │   └── utils.py                    # contains utility functions such as setting random seed and showing hyperparameters
        ├── trainer
        │   └── trainer_reward.py           # processes input arguments of a user for training
        ├── data_loader
        │   ├── ECData_loader_decoder.py
        │   └── MELD_loader_decoder.py
        ├── models                      
        │   ├── architecture_reward.py      # implements the forward function of the architecture
        │   ├── modules.py                  # loda dataset for dataloader
        │   ├── tf_decoder.py               # loda dataset for dataloader
        │   └── rewards
        │       ├── __init__.py
        │       ├── question.py
        │       ├── toxicity.py
        │       ├── sentiment.py
        │       └── semantic_similarity.py  
        └── eval_metric                     # pycocoeval
```

## How To Run

You can simply check if the model works correctly with the following command:
```
PYTHONPATH=src python3 run.py --arch_name $ARCHITECTURE
```
The above command will start learning the model on the `$ARCHITECTURE` with the specified parameters saved in `param.json`.