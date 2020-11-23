## Project Description

This is the final project for class Big Data Management and Analytics. Contributors include Minchao Zhu, Mo Han and Yibo Cheng. The project aims to implement a convolution neutral network without using pre-built library to do image classification.



## Project Outline

The project mainly has four steps to take and run

- read image data and preprocess images

- create network for training

- fit testing data

- analyze results, including accuracy, parameter effects

  

## Project Resource

Input dataset's structure is presented below

- dataset
  - training
    - cats (~4000 images)
    - dogs (~4000 images)
  - testing
    - cats (~1000 images)
    - dogs (~1000 images)

Dataset is downloaded from Kaggle website and link is presented here as well for reference.

```
https://www.kaggle.com/chetankv/dogs-cats-images
```



Submission folder's structure is also presented below

- project.ipynb
- dataset
- readme.md
- report.pdf

## How to Run

In order to run the project successfully in local environment, there are few extensions that are required to download, including tensor flow as it is used for image process, and necessary python extension.

Simply just open the python notebook and click run. The project will carry out the tasks including reading images, create network, train and fit model and present results. Please do not change the structure of project as program will read directory path for dataset and images.