# 20QsOverKG
This project partially contributes to the full outcome of Junyu Mao's master project at Univeristy College London, providing codes to reproduce the work. 


### Dependencies
Python 3.6
[Pytorch](https://pytorch.org) >= 1.0.0

<br>

### Prerequisites 

Clone the repository to local
```
git clone https://github.com/junyumao1996/QsOverKG.git
cd QsOverKG
```

Installing the requirements
```
pip3 install -r requirements.txt
```
<br>

### Datasets & Preprocessing
Run follows to download the raw datasets in src_data\
```
chmod +x download_data.sh
./download_data.sh
```
Once the datasets are download, running for preprocessing:
```
python process_datasets.py
```
<br>