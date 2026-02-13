# Clone the repository
```
# SHH
git clone --recursive git@github.com:Krist-Kul/Deep-Learning-Project.git
```
or
```
git clone --recursive https://github.com/Krist-Kul/Deep-Learning-Project.git
```
# Create conda environment
```batch
conda create -n traffic-object-detection python=3.10
conda activate traffic-object-detection
```
Then install libraries
```batch
pip install requirements.txt
```
# Run `import_data.py` to import your dataset from [Kaggle](https://www.kaggle.com/datasets/saumyapatel/traffic-vehicles-object-detection) by using this command 
```batch
python import_data.py
```
