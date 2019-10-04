# FakeNewsJudgement
## How to run the code
1. First, please install `Anaconda3` yourself. Then create the virtual environment by typing in:
```
conda env create -f environment.yaml
conda activate fnj
```
2. make directory for tensorboard event files, and then start the tensorboard
```
mkdir vis_log
nohup tensorboard --logdir vis_log --port 6008 &
```
3. Then run the whole experiment by simply run the script `run.sh`:
```
bash sh/run.sh
```
4. Watch the train process on the website localhost:6008
## Tips
* Please execute the commands under the root directory of the project.
* All the configurations should be configed in the file `run.sh`
