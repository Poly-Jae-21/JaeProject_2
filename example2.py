rewards_log = [[] for _ in range(3)]

for i in range(10):
    for j in range(3):
        reward = i+j
        rewards_log[j].append(reward)

print(rewards_log)
print([lst[-1] for lst in rewards_log])
