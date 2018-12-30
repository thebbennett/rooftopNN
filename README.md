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


## [Exploratory Data Analysis](https://github.com/thebbennett/rooftopNN/blob/master/Rooftop%20NN%20EDA.ipynb)
The exploratory data analysis of the 1,000+ image dataset, provided by Denver Open Data

## [DataGen](https://github.com/thebbennett/rooftopNN/blob/master/DataGen.ipynb)
The code used for the data generator
The base of my data generator takes an (1000, 1000) px  image and it’s corresponding mask and:
* randomly rotates the images between 0 and 360 degrees
* randomly crops the images to  size (256,256). 
* randomly flips the images along the vertical axis
* randomly rotates the images an additional 90 degrees

## [Model](https://github.com/thebbennett/rooftopNN/blob/master/model.py)
The model’s architecture is based off of the U-NET architecture U-NET is a artificial neural network based on ConvNets that is widely used for image segmentation work. The U-NET architecture which features a set of encoding layers then decoding layers. Each encoding chunk, as shown by the set of three lines in each row below, has a convolution block followed by a maxpool downsampling. The decoding section features upsampling and concatenation followed by more convolution. 

The modified model architecture I chose for this project is based off of [Fabio Sancinetti’s U-NET ConvNet for CT-Scan segmentation](https://medium.com/@fabio.sancinetti/u-net-convnet-for-ct-scan-segmentation-6cc0d465eed3). 






