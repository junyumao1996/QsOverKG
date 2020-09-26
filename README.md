# 20QsOverKG
### Overview
This project partially contributes to the full outcome of Junyu Mao's master project at Univeristy College London, "*Learning to Ask over Knowledge Graph: Reinforcement Learning for Game-Based Interactive Knowledge Acquisition*", providing codes to reproduce the work.
#### Learning-to-Ask-and-Balance (LAB) Framework
<img src="https://i.imgur.com/rC1wsXX.jpg" width="388">

#### RL-based Information Seeking (IS)
<img src="https://i.imgur.com/LGlma6I.png" width="488">

## Dependencies
Python 3.6
<br />
[Pytorch](https://pytorch.org) >= 1.0.0

<br>


### License
MIT License

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

### KG Datasets Preprocessing
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

## Run
For the result of a complete game loop using (LAB-WOLPDPG + LAB-ComplEx + Balancer), run the following in the root directory
```
python main.py
```
<br>


## References
Some repositories are used with local modifications for our study:
* Deep RL tutorials <https://github.com/qfettes/DeepRL-Tutorials>
* Wolpteringer training <https://github.com/ChangyWen/wolpertinger_ddpg>
* Canonical Tensor Decomposition for KBC <https://github.com/facebookresearch/kbc>