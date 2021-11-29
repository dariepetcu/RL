# RL_Practical
reinforcement learning practical. /git gud for instant homework completion

# Questions for the lab:
* Do we want **one** multi-armed bandit or *multiple*?
  
    Run N k-armed bandits to get N samples of data
* Is it okay to use (at least as inspiration) code from AI2 assignments?
  
    Code written by us fine, code from uni not
* How complex do we want our environment to be?
  
    Just k probabilities stored in array is fine
* Is the bandit the bandit, or is it the agent that chooses?
  
    Bandit = slot machine. k armed = k machines. One agent interacts with all k arms
* How to select epsilon for eps-greedy?
  
    Choose best value from trial and error
* What the fuck is slide 17
    
    Probability to be greedy

* How to select initial value when not using optimistic initial values?

    They're just 0

* Gaussian distr: do we need to have different values for both the mean and SD
  or can we just use different means and one SD
  
    use diff means AND SD. otherwise "try and see what happens ;)"


* What to do after using optimistic initial values? greedy or?

  yes. greedy, and if time allows for it, try other stuff as well

* Do we need the incremental update of sample-average method for anything apart from optimization?

  incremental is more optimized

* UCB: What is Na(t)?

  number of times a was chosen after t timesteps

* optimistic initial values are set higher than the highest possible reward, manually. the value itself depends on the
  reward implementation
  
* fill between: matplotlib to show some statisticsy stuff