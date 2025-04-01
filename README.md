# Learn-to-Rank Fairness Evalution

## Instllation and Running
To run the project, it requires `python 3.10`. It can be downloaded from https://www.python.org/downloads/release/python-3100/ or a kernel with `python 3.10` can be used. 

Running the first code block within the `main.ipynb` file or shell command below will install all dependencies for running the jyupiter notebook with the main pipepline. 

```shell
pip install "numpy<2.0" python-terrier==0.12.1 nltk scikit-learn lightgbm fastrank tensorflow==2.11 keras LambdaRankNN
```

## Code structure
The models can each be found in their respective file within the `models` directory. 
The main pipeline to be run can be found in the `main.ipynb` notebook. Running the code blocks in order will perform training and evaluation for all models on a single dataset. 
All other utility functions such as fairness evaluation, dataset handling and computing coefficient can be found in their respective python file in the source directory.

## Collaborators
- Alexandru Turcu - acturcu
- Sebastian Manda Alexandru - SebastianManda
- Pepijn de Kruijff - pepijndk
- Daniel Chou Rainho - daniel-cho-rainho
