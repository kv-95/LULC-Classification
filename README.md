<============================== READ ME ==============================>

This work creates a model for the LULC Classification using the novel [EuroSAT](https://github.com/phelber/EuroSAT) dataset. 
A sample model is included in the "Model/" directory.

TO CREATE A NEW MODEL:

1. SPECIFY THE PATH TO DATAFROLDER IN "Dataset.py" > data_dir.
2. RUN "classify.py"

TO CHECK AN IMAGE FOR ITS CLASSIFICATION (WITH EXSITING MODEL):

1. Update line 10 test_path="path\to\image\image.jpg"
2. RUN "check.py" (make sure you have model saved in a directory "Model/")


### References

1. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.
2. Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. Patrick Helber, Benjamin Bischke, Andreas Dengel. 2018 IEEE International Geoscience and Remote Sensing Symposium, 2018.
