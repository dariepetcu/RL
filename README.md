# RL Practical - Final Report

### Farrukh Baratov (s3927083)
### Darie Petcu (s3990044)

## Requirements
Python 3.10 is used for this program. The program's directory must also have a folder called 'plots',
which contains a folder 'tuning' (for the tuning.py functions to work).

The following packages are also required:
* matplotlib
* numpy

The following built-in packages are required:
* random
* math
* sys
* types
* enum

##How to run the code

The following command can be used to run the code (at least one should work):
```
python main.py
python3 main.py
```

This runs the main.py, which trains each possible RL-agent against both of the hardcoded algorithms
over 100,000 epochs and saves a graph to the directory called /plots.
##Included files

* main.py: main file, creates agent/game, calls training and plots graphs.
* agent.py: contains agent
* env.py: Contains ConnectX environment.
* play.py: contains playing/training environments.
* plot.py: contains functions used for plotting results.
* tuning.py: contains functions used for tuning hyperparameters. Not a necessary file to run the code as-is.