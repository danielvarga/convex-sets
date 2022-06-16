import numpy as np
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

n = 10

sides = []
for i in range(4):
    a = np.random.uniform(size=n-2).tolist() + [0, 1]
    sides.append(sorted(a))

sides = np.array(sides)

values = sides.copy()
values[1] = 1 - values[1]
values[2] = 1 - values[2]

# (a,b,c) for line ax+by+c=0 through p1 and p2.
# (y-y1)*(x2-x1) = (y2-y1)(x-x1), that is,
# y*(x2-x1) + (-y2+y1)*x + (x1-x2)*y1+x1*(y2-y1) = 0

def line_through(p1, p2):
    x1, y1 = p1[..., 0], p1[..., 1]
    x2, y2 = p2[..., 0], p2[..., 1]
    return x2 - x1, y1-y2, (x1-x2)*y1+x1*(y2-y1)


# l_i are (a, b, c) tuples, possibly vectorized arrays.
# a1x+b1y+c1=0
# a2x+a2y+c2=0
# https://www.cuemath.com/algebra/linear-equations-in-two-variables/
def intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    x = (c1 * a2 - c2 * a1) / d
    y = (b1 * c2 - b2 * c1) / d
    return x, y


# line (a, 0)-(b, 1) intersected with lines
# (0, left[i])-(1, right[i]), projected to the second coordinate
def intersection_projections(a, b, left, right):
    p_left = np.stack([np.zeros_like(left), left]).T
    p_right = np.stack([np.ones_like(right), right]).T
    p1 = np.array([a, 0])
    p2 = np.array([b, 1])
    vert = line_through(p1, p2)
    hors = line_through(p_left, p_right)
    x, y = intersect(vert, hors)
    return p_left, p_right, x, y


def test_intersection_projections():
    fig = plt.figure()
    a = 0.3
    b = 0.7
    left = np.sort(np.random.uniform(size=10))
    right = np.sort(np.random.uniform(size=10))
    p_left, p_right, x, y = intersection_projections(a, b, left, right)
    plt.scatter(x, y)
    plt.scatter(p_left[:, 0], p_left[:, 1])
    plt.scatter(p_right[:, 0], p_right[:, 1])
    plt.show()


# test_intersection_projections() ; exit()

left, right = sides[0], sides[2]
down, up = sides[1], sides[3]
down_v, up_v = values[1], values[3]

fig = plt.figure()
ax  = fig.add_subplot(111, projection = '3d')

for i in range(n):
    a, b = down[i], up[i]
    p_left, p_right, x, y = intersection_projections(a, b, left, right)
    level = down_v * (1 - x) + up_v * x
    ax.scatter(x, y, level, s=10)
plt.show()
exit()


for direction in range(2):
    left, right = sides[direction], sides[direction+2]
    left_v, right_v = values[direction], values[direction+2]
    if direction == 0:
        for i in range(n):
            plt.plot([left[i], right[i]], [0, 1], [left_v[i], right_v[i]])
    else:
        for i in range(n):
            plt.plot([0, 1], [left[i], right[i]], [left_v[i], right_v[i]])

plt.show()