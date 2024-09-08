import matplotlib.pyplot as plt

times1 = []
times2 = []


fig, ax = plt.subplots()

ax.set(xlim=(-30,1050), ylim=(-220,250))
ax.set_xlabel('Number of training episodes')
ax.set_ylabel('Training loss')
plt.show()