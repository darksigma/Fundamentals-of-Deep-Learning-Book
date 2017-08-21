import random
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

step_range = 10
momentum_range = [0.1, 0.5, 0.9, 0.99]

step_choices = range(-1 * step_range, step_range + 1)
rand_walk = [random.choice(step_choices) for x in xrange(100)]


x = range(len(rand_walk))
zeros = [0 for i in x]

import numpy as np
yrange =  1.5 * np.max(rand_walk)

fig = plt.figure(1)
gs = gridspec.GridSpec(3, 4)
ax = plt.subplot(gs[0, 1:3])
ax.set_title("No Momentum")
plt.xlabel("steps")
plt.plot(x, rand_walk, 'b', x, zeros, 'k')
plt.ylim((-yrange, yrange))


momentum = momentum_range[0]
momentum_rand_walk = [random.choice(step_choices)]

for i in xrange(len(rand_walk) - 1):
	prev = momentum_rand_walk[-1]
	momentum_rand_walk.append(momentum * prev + (1 - momentum) * random.choice(step_choices))

ax = plt.subplot(gs[1,:2])
ax.set_title("Momentum = %s" % momentum_range[0])
plt.plot(x, momentum_rand_walk, 'r', x, zeros, 'k')
plt.ylim((-yrange, yrange))

momentum = momentum_range[1]
momentum_rand_walk = [random.choice(step_choices)]

for i in xrange(len(rand_walk) - 1):
	prev = momentum_rand_walk[-1]
	momentum_rand_walk.append(momentum * prev + (1 - momentum) * random.choice(step_choices))

ax = plt.subplot(gs[1,2:])
ax.set_title("Momentum = %s" % momentum_range[1])
plt.plot(x, momentum_rand_walk, 'r', x, zeros, 'k')
plt.ylim((-yrange, yrange))

momentum = momentum_range[2]
momentum_rand_walk = [random.choice(step_choices)]

for i in xrange(len(rand_walk) - 1):
	prev = momentum_rand_walk[-1]
	momentum_rand_walk.append(momentum * prev + (1 - momentum) * random.choice(step_choices))

ax = plt.subplot(gs[2,:2])
ax.set_title("Momentum = %s" % momentum_range[2])
plt.plot(x, momentum_rand_walk, 'r', x, zeros, 'k')
plt.ylim((-yrange, yrange))

momentum = momentum_range[3]
momentum_rand_walk = [random.choice(step_choices)]

for i in xrange(len(rand_walk) - 1):
	prev = momentum_rand_walk[-1]
	momentum_rand_walk.append(momentum * prev + (1 - momentum) * random.choice(step_choices))

ax = plt.subplot(gs[2,2:])
ax.set_title("Momentum = %s" % momentum_range[3])
plt.plot(x, momentum_rand_walk, 'r', x, zeros, 'k')
plt.ylim((-yrange, yrange))

fig.tight_layout()

plt.show()
