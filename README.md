# RL Practical - Multi-Armed Bandit Assignment

### Farrukh Baratov (s3927083)
### Darie Petcu (s3990044)

##How to run the code
`Agent.py`, `Problem.py`, and `main.py` all need to be in the same working directory. For saving the plots,
an extra directory called `plots` is required, with the subdirectories `avg`, `acc`, and `tuning` is required.

By default, the program runs with the Gaussian distribution and standard hyperparameters, going through all
action selection algorithms one by one. The problem has 7 arms by default Each algorithm is tested for 1000
timesteps, and the experiment is run 1000 times. The terminal will display a message whenever one of the 
algorithms is done with all iterations. After the process has finished, the plots for average rewards
and accuracies will be saved in the `/acc` and `/avg` subdirectories of the working directory.

The function `run_tuning` plots the results of an algorithm with different values for its hyperparameter. Plots
will be saved in the directory `plots/tuning`. This function is not called by default.

The following command runs the code:
```
python3 main.py
```