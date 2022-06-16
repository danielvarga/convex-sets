import numpy as np
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx

import matplotlib.pyplot as plt

n = 30

np.random.seed(2)

sides = []
for i in range(4):
    a = np.random.uniform(size=n-2).tolist() + [0, 1]
    sides.append(sorted(a))

sides = np.array(sides)

values = sides.copy()
values[1] = 1 - values[1]
values[2] = 1 - values[2]


def test_manifold():
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection = '3d')
    for direction in range(2):
        left, right = sides[direction], sides[direction+2]
        left_v, right_v = values[direction], values[direction+2]
        if direction == 0:
            for i in range(n):
                ax.plot([left[i], right[i]], [0, 1], [left_v[i], right_v[i]])
        else:
            for i in range(n):
                ax.plot([1, 0], [left[i], right[i]], [left_v[i], right_v[i]])
    plt.show()

# test_manifold() ; exit()


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

fig = plt.figure()
ax  = fig.add_subplot(111, projection = '3d')
color_map = plt.get_cmap('viridis')
cNorm = matplotlib.colors.Normalize(vmin=0, vmax=2 * n)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_map)

left, right = sides[0], sides[2]
down, up = sides[1], sides[3]
down_v, up_v = values[1], values[3]

# ax.scatter(down, np.zeros_like(down), down_v, marker='v')
# ax.scatter(up, np.ones_like(up), up_v, marker='v')


horizontal_levels = []
for i in range(n):
    a, b = down[i], up[i]
    p_left, p_right, x, y = intersection_projections(a, b, left, right)
    level = down_v[i] * (1 - y) + up_v[i] * y
    horizontal_levels.append(level)
    ax.scatter(x, y, level, s=50, color=scalarMap.to_rgba(i), marker='.')
horizontal_levels = np.array(horizontal_levels)

left, right = sides[1], sides[3]
down, up = sides[0], sides[2]
down_v, up_v = values[0], values[2]

vertical_levels = []
for i in range(n):
    a, b = down[i], up[i]
    p_left, p_right, x, y = intersection_projections(a, b, left, right)
    level = down_v[i] * (1 - y) + up_v[i] * y
    level = 1 - level
    vertical_levels.append(level)
    ax.scatter(y, x, level, s=50, color=scalarMap.to_rgba(n + i), marker='x')
vertical_levels = np.array(vertical_levels).T

plt.show()

print((horizontal_levels - vertical_levels > 0).astype(int))
