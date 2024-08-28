# Aspect-Category-Sentiment-Analysis-using-Real-time-Big-Data-Processing
# Overview
*  Training a model for Aspect Category Sentiment Analysis task on Hotel data
*  Build a real-time Spark dashboard interface system

## Table of Contents

- [Model](#model)
- [Dashboard](#daashboard)
- 
# Model
Our ViMACSA dataset comprises 4,876 documents and 14,000 images. Each document is accompanied by up to 7 images. This dataset is constructed with the goal of recognizing both explicit aspects and implicit aspects in the document.

<p align="left">
  <img src="images/model.png" />
</p>

## Distributed Data Parallel
<p align="left">
  <img src="images/ddp.png" />
</p>

## Results
![Results on the F1 score of the models.](images/results_all_model.png)
*Table 1. Results on the F1 score of the models.*


![Detailed results of the emotion labels from the Bert-multilingual model.](images/result_bestt.png)
*Table 2. Detailed results of the emotion labels from the Bert-multilingual model.*

![Results on the F1 score of the models.](images/time.png)
*Training time of each model.*


# Dashboard
![Architecture of system](images/archo.png)                   
*Architecture of system*



