# Identifying Area of Rooftops in Denver, CO, for Solar Policy 

Independent capstone project to Springboard's Data Science program. 

Given satellite imagery of Denver, we can train a convolutional neural network to correctly identify rooftop area. 

* Label 75+ images of rooftops with the tool provided generously by Dr. Tony Szedlak
* Build a data generator to apply random cropping of size (256,256) of the (800,800) images
* Build a CNN 
* Train the CNN given the generated data 
* Fine tune the model’s hyperparameters to increase model accuracy  
* Predict given all 1,800+ satellite images for Denver, CO

## Data Description 
The City and County of Denver generously provides open source data via [https://www.denvergov.org/opendata](https://www.denvergov.org/opendata). For this project, I am using their 2004 dataset of satellite imagery of Denver County. These data consist of almost 2,500 JP2 files. 

From the Open Data website:
* Color, 6 inch resolution aerial photography for the City and County of Denver acquired in 2004. True-ortho correction is applied to all structures 4 stories and above in designated areas including all bridges and highway overpasses.
* Coverage includes a 359 square mile area encompassing the City and County of Denver, City of Glendale, City of Littleton, and the Denver Water Service Area. This project does not include DIA.
* Spatial reference is NAD 1983 HARN StatePlane Colorado Central FIPS 0502.
* For the benefit of the analysis, I converted these JP2 files into JPEGs using XnConverter for Linux. 



## [Exploratory Data Analysis] (https://github.com/thebbennett/rooftopNN/blob/master/Rooftop%20NN%20EDA.ipynb)
The exploratory data analysis of the 1,000+ image dataset, provided by Denver Open Data

## [DataGen ] (https://github.com/thebbennett/rooftopNN/blob/master/DataGen.ipynb)
The code used for the data generator




