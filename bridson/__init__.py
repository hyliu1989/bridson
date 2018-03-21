from random import random
from math import cos, sin, floor, sqrt, pi, ceil
import numpy as np
import itertools


def getEuclideanDistance(a, b):
    # dx = a[0] - b[0]
    # dy = a[1] - b[1]
    # return sqrt(dx * dx + dy * dy)
    return np.linalg.norm(a-b)  # return an N dimensional Euclidean distance


def poisson_disc_samples(dims, r, k=5, distance=getEuclideanDistance, random=random):
    """
    dims: the dimensions for all sides of the n-dimensional box (e.g. (height, width) = dims in 2D)
          the units are the same as that for `r`. Can be, for example, meter or cm.
    r:    least distance between each point
    """
    n_dimensions = len(dims)
    assert n_dimensions <= 3
    cellsize = r / sqrt(n_dimensions)
    grid_shape = np.ceil(np.array(dims)/cellsize).astype(np.int)
    grid = np.empty(dims, dtype=object)

    def getGridCoords(p):
        return tuple(np.floor(p/cellsize).astype(np.int))

    def fits(p, grid_coord):
        ranges = []
        for i, coord_element in enumerate(grid_coord):
            ranges.append(range(max(coord_element - 2, 0), min(coord_element + 3, grid_shape[i])))
        ranges = tuple(ranges)
        for coord in itertools.product(*ranges):
            g = grid[coord]
            if g is None:
                continue
            if distance(p, g) <= r:
                return False
        return True

    if n_dimensions == 1:
        getUniformDistributionOnAnnulus = getUniformDistributionOnAnnulus1D
    elif n_dimensions == 2:
        getUniformDistributionOnAnnulus = getUniformDistributionOnAnnulus2D
    elif n_dimensions == 3:
        getUniformDistributionOnAnnulus = getUniformDistributionOnAnnulus3D

    point_p = np.array([s * random() for s in grid_shape])
    queue = [point_p]
    grid_coord = getGridCoords(point_p)
    grid[grid_coord] = point_p

    while queue:
        qi = int(random() * len(queue))
        point_q = queue[qi]
        queue[qi] = queue[-1]
        queue.pop()
        for _ in range(k):
            point_p = point_q + getUniformDistributionOnAnnulus(r)
            inside = np.all(0 <= point_p) and np.all(point_p < dims)
            if not inside:
                continue
            grid_coord = getGridCoords(point_p)
            if not fits(point_p, grid_coord):
                continue
            queue.append(point_p)
            grid[grid_coord] = point_p
    return [e for e in grid.ravel() if e is not None]



def getUniformDistributionOnAnnulus1D(r):
    """Use CDF to obtain the uniform distribution on 1D annulus ranging from r to 2r.
    """
    sign = -1 if random() < 0.5 else 1
    radius = r * (random()+1)
    return radius * sign

def getUniformDistributionOnAnnulus2D(r):
    """Use CDF and polar coordinates to obtain the uniform distribution on 2D annulus ranging from r to 2r.
    """
    phi = 2*np.pi*random()
    radius = r * np.sqrt(3*random()+1)
    return radius * np.array([np.cos(phi), np.sin(phi)])

def getUniformDistributionOnAnnulus3D(r):
    """Use CDF and spherical coordinates to obtain the uniform distribution on 3D annulus ranging from r to 2r.
    """
    theta = np.arccos(2*random()-1)
    phi = 2*np.pi*random()
    radius = r * np.cbrt(7*random()+1)
    return radius * np.array([np.cos(theta), np.sin(theta)*np.sin(phi), np.sin(theta)*np.cos(phi)])
