# project-fa19
CS 170 Fall 2019 Project

### Set up environment
1. Install docker
2. 
```bash
docker run -it ziyaointl/traveling-rao:0.0.1 bash
```
3. Place this folder inside the docker container, either through git clone or manually copying

### Run on single input
Note: Please perform the following operations inside the docker environment
1.
```bash
cd CS170-Traveling-Rao
python main.py <input> # e.g. python main.py 14_50
```

### Run on multiple inputs
1. On google cloud, create a new kubernetes cluster
2. Initialize helm
```bash
helm init
```
3. Install dask using the dask folder contained in the helm directory, including values.yaml
```bash
helm install -f values.yaml ./dask/dask -n dask
```
4. Lookup the scheduler ip
```bash
helm status dask
```
Note: Please perform the following operations inside the docker environment
5. Replace the scheduler ip in main.py
6. Launch! The program will automatically detect unfinished outputs and submit them to the scheduler.
```bash
python main.py
```
