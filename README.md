# 20QsOverKG
### Overview
This project partially contributes to the full outcome of Junyu Mao's master project, "Learning to Ask over Knowledge Graph: Reinforcement Learning for Game-Based Interactive Knowledge Acquisition", at Univeristy College London, providing codes to reproduce the work.
#### Learning-to-Ask-and-Balance
![](https://i.imgur.com/rC1wsXX.jpg=200x)

#### RL-based Information Seeking (IS)
![](https://i.imgur.com/LGlma6I.png)

### Dependencies
Python 3.6 
<br />
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
Run follows to download the raw datasets in ./src_data folder and run
```
chmod +x download_data.sh
./download_data.sh
```
Once the datasets are download, running for preprocessing
```
python process_datasets.py
```
<br>