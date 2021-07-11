<============================== READ ME ==============================>

This work creates a model for the LULC Classification using the novel [EuroSAT](https://github.com/phelber/EuroSAT) dataset. 
A sample model is included in the "Model/" directory.

TO CREATE A NEW MODEL:

1. SPECIFY THE PATH TO DATAFROLDER IN "Dataset.py" > data_dir.
2. RUN "classify.py"

TO CHECK AN IMAGE FOR ITS CLASSIFICATION (WITH EXSITING MODEL):

1. Update line 10 test_path="path\to\image\image.jpg"
2. RUN "check.py" (make sure you have model saved in a directory "Model/")
