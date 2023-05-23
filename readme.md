## Deep learning for segmentation on SOHO images

### ABSTRACT
Solar activity has influence on many aspects, such as climate or technology. Therefore, it is important to monitor and identify phenomena occurring on the Sun in order to predict and better understand their impact on us. This work deals with the segmentation of coronal holes and active regions on images captured by the SOHO spacecraft, which has been monitoring the Sun since 1996. For segmentation we used a SCSS-Net convolutional neural network. The results of this work comprise new annotations of these two phenomena and their visualization, which can be used for further research on solar activity.    

These videos shows SCSS-net performance on the SOHO dataset for years 1996 - 2021
- https://youtu.be/IszyBdDoexU
- https://youtu.be/rKJZ2aSppto


### STRUCTURE
- [prerequisites](prerequisites/) contains steps for downloading data and installing necessary libraries
- after downloading images provided in [prerequisites notebook](prerequisites/prerequisites.ipynb), [data folder](data/) should contain images for this project
- [modeling folder](modeling/) contains notebooks used for analysis regarding segmentation of active regions and coronal holes
- [src folder](src/) contains SCSS-net convolutional neural network from https://arxiv.org/pdf/2109.10834.pdf 
- [preprocessing folder](preprocessing/) contains code and scripts to preprocess data. There is no need for user to run those scripts again.
- [webapp folder](webapp/) contains files for simple web based interface with user. In this web interface, user can browse images and view segmentations of coronal holes or active regions on those images. To run webapp, download data provided in [prerequisites notebook](prerequisites/prerequisites.ipynb), then navigate to webapp directory in terminal and run following line:
```console
Streamlit run webapp.py
``` 
Preview of webapp can be seen on image below

![webapp folder](figures/webapp-peak.png)
