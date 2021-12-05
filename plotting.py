import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
plt.plot(rewards, label="UCB")
plt.legend(bbox_to_anchor=(1.3, 0.5))
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Average Gradient Bandit Rewards after "
      + str(T) + " Episodes")
plt.show()