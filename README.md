Welcome to use our code! Please follow the steps below to correctly download and set up your coding environment.

# Step 0:

Please create your virtual environment based on environment.yml file.
Ensure that your working environment includes bedtools (version: v2.30.0) and meme (version: 5.5.8).

# Step 1: Download the Compressed Package
Please download our compressed package from the following link:

Download Link

# Step 2: Extract the Files
Extract the downloaded compressed file into the current folder. After extraction, you should see a folder containing multiple subdirectories and files.

# Step 3: Download Necessary Public Data
Enter the public_data folder, where you will find a readme file that contains instructions on how to obtain the necessary data. Please read carefully and follow the instructions to download the required files.

# Step 4: Ensure the Directory Structure is Consistent
After downloading all the necessary data, please check whether your directory structure matches the following template:
```
your_project_folder/
│
├── public_data/
│   └── ...
│   
├── Data/
│   └── ...
│
├── GM12878/
│   ├── bulk.bigwig
│   └── sc.bw
│
├── mouse/
│   └── bulk.bigwig
│
├── K562/
│   ├── bulk.bigwig
│   └── sc.bw
│
├── TableS1.xlsx
│
└──  other python files
```
# Step 5:Pipeline Execution in Jupyter Notebook

We will execute the data processing pipeline step-by-step using Python scripts.

In the following scripts, there are four parameters that you can adjust: cell_line, mode, bw_type, and species.

Use the cell_line parameter to select the desired cell line for processing. 
Available choices are GM12878, K562, and Mouse CD8 T cells.

Use the mode parameter to select the desired type for processing. 
Available choices are E (Enhancer), P (Promoter), and EP (Enhancer and Promoter). If the species is set to mouse, only P can be selected.

Use the bw_type parameter to select the data source for processing, choosing between bulk and single-cell.
Available choices are bulk and sc(single cell). If the species is set to mouse, only bulk can be selected.

Use the species parameter to select whether to process human cells or mouse cells.
Available choices are human and mouse.

#### If in Step 1 the species is set to "Mouse CD8 T cells", please first run Step 1 with cell_line=GM12878, mode=P, and species=human.

## Step 1: Calculate Clustering Coefficient
This script calculates the clustering coefficient using the specified file path, cell line, and mode.

```python script example
!python Calculate_clustering_coefficient.py --file_path TableS1.xlsx --cell_line GM12878 --mode P --species human
```

## Step 2: Feature Extraction
Next, we extract features from the data using the specified parameters.

```python script example
!python Feature_Extraction.py --work_dic ./ --cell_line GM12878 --mode P --bw_type sc --file_path ./TableS1.xlsx --species human
```

## Step 3: Data Splitting
Now, split the data into training, testing, and validation sets.

```python script example
!python train_test_val_split.py --xlsx_path TableS1.xlsx --cell_line GM12878 --mode P --bw_type sc
```

## Step 4: Model Training
With the data prepared, we can now train the model.

```python script example
!python train.py --cell_line GM12878 --mode P --bw_type sc
```

## Step 5: Generate Picture
Finally, visualize the ROCAUC/PRAUC/Confusion Matrix results by generating pictures.

```python script example
!python picture.py --cell_line GM12878 --mode P --bw_type sc
```

## Step 6:Predict
To make predictions using the `predict.py` script, you can run the following command:

```
!python predict.py \
    --predict_cell_line 'Mouse CD8 T cells' \
    --mode P \
    --bw_type bulk \
    --model_cell_line GM12878 \
    --model_path path/to/your/model.pth \
    --predict_file "path/to/your/file.csv"
```
tips:you could use "Mouse CD8 T cells/P/traindata.csv" as predict_file as test

