# **NLP Pipeline**

This repository contains an text machine learning pipeline and some of the projects i've worked on, either for fun/education or competition on Kaggle. \
Each project have it's own readme containing information about the specific problematics of each. 

I train the models locally on my pc using a Nvidia 1080 GPU. 
## **Data**

The data is not in the repository directly if you want to launch a model on one the projects in here you must download the data and change the config file in the task folder to be adequate. \
Links to the datasets are in the tasks README's.
## **Projects**
---
The projects folder contains the specific code about each project :
 * config.py file containing  most of the parameters necessary to train a model.
 * feature_eng.py that contain the specific pre-processing functions you want to perform on the text for training, validation and testing

### **What if i want to add a new project ?**
To add a new project you'll to create a few things : 
* a new folder in the projects/ folder containing a \_\_init\_\_.py file, a config.py file and a feature_eng.py file
* copy and paste the content of another config.py file and change the information to be adequate with your task
* Add the functions you require to the feature_eng.py file

To start training a model on any task use this command in terminal :
```
python -m run --project=tweet_disaster
```
You can replace the **tweet_disaster** with any folder in projects/.
Default parameters train for **5** folds using a **DISTILBERT** model.
You can change these parameters as such :
```
python -m run --folds=8 --project=commonlit" --model_name=BERT
```

The parameters can take different values :
* **project** : The project you want to train a model on, atm you can train a model on the aerial_cactus task, melanoma, blindness_detection & leaf_disease projects.
* **folds** : this parameter determine the number of folds to create into the dataset. If you choose 5 for example, the dataset will be divided in 5, train a model on 4 folds and validate on the last (folds 0, 1, 2, and 3 for training and 4 for validation. Then, it'll train on folds 0, 1, 2, 4 and validate on 3 etc...).
* **model_name** : You can choose any model that is in the models/ folder, name must be typed in MAJ like in the example above.

## **To do** 
---
* Configure the inference file
* Add code the allow for Q&A projects
* Add scheduler
* Add more models
* Add more loss functions available
* Add metrics
* Add logger
* Add notebooks (for model evaluation - EDA - hyperparameter optimization etc...)