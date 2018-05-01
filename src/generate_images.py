import os
import random
import time
import copy
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random


SMALL_SIZE = 0
BIG_SIZE = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
####

class Point(object):
    def __init__(self, *coordinate):
        coordinate = coordinate[0]
        self.x = float(coordinate[0])
        self.y = float(coordinate[1])


class FuzzySet(object):
    def __init__(self, name):
        self.name = name


class Triangle(FuzzySet):
    """
           _
          / \
         / | \
        /  |  \
       /   |   \
     _/    |    \_
      |    |    |
      a    b    c
    """
    def __init__(self, name, *points):
        super(Triangle, self).__init__(name)
        self.a = Point(points[0])
        self.b = Point(points[1])
        self.c = Point(points[2])
        assert self.a.y == self.c.y
        assert self.a.x < self.b.x
        assert self.b.x < self.c.x
        self.kab = (self.b.y - self.a.y) / (self.b.x - self.a.x)
        self.kbc = (self.c.y - self.b.y) / (self.c.x - self.b.x)

    def membership(self, x):
        x = float(x)
        if x < self.a.x:
            return 0
        elif x < self.b.x:
            return self.kab * (x - self.a.x) + self.a.y
        elif x < self.c.x:
            return self.kbc * (x - self.b.x) + self.b.y
        else:
            return 0


class Trapezoid(FuzzySet):
    """
           _____
          /     \
         /|     |\
        / |     | \
       /  |     |  \
     _/   |     |   \_
      |   |     |   |
      a   b     c   d
    """
    def __init__(self, name, *points):
        super(Trapezoid, self).__init__(name)
        self.a = Point(points[0])
        self.b = Point(points[1])
        self.c = Point(points[2])
        self.d = Point(points[3])
        assert self.a.y == self.d.y
        assert self.b.y == self.c.y
        assert self.a.x < self.b.x
        assert self.b.x < self.c.x
        assert self.c.x < self.d.x
        self.kab = (self.b.y - self.a.y) / (self.b.x - self.a.x)
        self.kcd = (self.d.y - self.c.y) / (self.d.x - self.c.x)

    def membership(self, x):
        x = float(x)
        if x < self.a.x:
            return 0
        elif x < self.b.x:
            return self.kab * (x - self.a.x) + self.a.y
        elif x < self.c.x:
            return self.b.y
        elif x < self.d.x:
            return self.kcd * (x - self.c.x) + self.c.y
        else:
            return 0


class LeftSkewTrapezoid(FuzzySet):
    """
       _____
      |     \
      |     |\
      |     | \
      |     |  \
     _|     |   \_
      |     |   |
      a     b   c
    """
    def __init__(self, name, *points):
        super(LeftSkewTrapezoid, self).__init__(name)
        self.a = Point(points[0])  # lower point
        self.b = Point(points[1])
        self.c = Point(points[2])
        assert self.a.y == self.c.y
        assert self.a.x < self.b.x
        assert self.b.x < self.c.x
        self.kbc = (self.c.y - self.b.y) / (self.c.x - self.b.x)

    def membership(self, x):
        x = float(x)
        if x < self.a.x:
            return 0
        elif x < self.b.x:
            return self.b.y
        elif x < self.c.x:
            return self.kbc * (x - self.b.x) + self.b.y
        else:
            return 0


class RightSkewTrapezoid(FuzzySet):
    """
           _____
          /     |
         /|     |
        / |     |
       /  |     |
     _/   |     |_
      |   |     |
      a   b     c
    """
    def __init__(self, name, *points):
        super(RightSkewTrapezoid, self).__init__(name)
        self.a = Point(points[0])
        self.b = Point(points[1])
        self.c = Point(points[2])  # lower point
        assert self.a.y == self.c.y
        assert self.a.x < self.b.x
        assert self.b.x < self.c.x
        self.kab = (self.b.y - self.a.y) / (self.b.x - self.a.x)

    def membership(self, x):
        x = float(x)
        if x < self.a.x:
            return 0
        elif x < self.b.x:
            return self.kab * (x - self.a.x) + self.a.y
        elif x < self.c.x:
            return self.b.y
        else:
            return 0
####
shape_list = ['Circle', 'Ellipse', 'Triangle', 'Square', 'Rectangle']

feature_list = [
    'Thinness',
    'Extent',
    'Corners'
]

fset_routes = {
    'TRI': Triangle,
    'TPZ': Trapezoid,
    'LST': LeftSkewTrapezoid,
    'RST': RightSkewTrapezoid
}

devel_image_dir = '/Users/dev/Shape/data/'
train_image_dir = '/Users/dev/Shape/data/'
train_feature_dir = '/Users/dev/Shape/data/'
test_image_dir = '/Users/dev/Shape/data/'
sketch_image_dir = '/Users/dev/Shape/data/sketch_image'


class RandomDegree(object):
    def __init__(self):
        self.counter = 0
        self.sequence = list(range(360))
        random.shuffle(self.sequence)

    def next(self):
        val = self.sequence[self.counter]
        self.counter += 1
        if self.counter == len(self.sequence):
            random.shuffle(self.sequence)
            self.counter = 0
        return val

####


def create_img():
    img = np.ones((800, 800, 3), np.uint8)
    img[:,:] = (255, 255, 255)
    return img


def sizable_radius(size):
    if size == SMALL_SIZE:
        return random.randrange(100, 150)
    elif size == BIG_SIZE:
        return random.randrange(150, 200)
    else:
        raise Exception('Unknown size.')


def safe_rand_center(width_span, height_span, width=800, height=800):
    margin = max(int(width_span * 0.63), int(height_span * 0.63))
    center_x = random.randrange(margin + width_span, width - margin - width_span)
    center_y = random.randrange(margin + height_span, height - margin - height_span)
    return center_x, center_y


def draw_circle(img, size):
    radius = sizable_radius(size)
    center_x, center_y = safe_rand_center(radius, radius)
    cv2.circle(img, (center_x, center_y), radius, BLACK, 3)
    return img


def draw_ellipse(img, size):
    long_radius = sizable_radius(size)
    short_radius = int(long_radius * random.uniform(0.5, 0.8))
    center_x, center_y = safe_rand_center(long_radius, long_radius)
    cv2.ellipse(img, (center_x, center_y), (long_radius, short_radius), 0, 0, 360, BLACK, 3)
    return img


def draw_triangle(img, size):
    long_radius = sizable_radius(size)
    short_radius = int(long_radius * random.uniform(0.5, 0.8))
    center_x, center_y = safe_rand_center(long_radius, short_radius)
    lb_pt = [random.uniform(center_x - long_radius, center_x - long_radius + int(0.63 * long_radius)), center_y + short_radius]
    rb_pt = [random.uniform(center_x + long_radius - int(0.63 * long_radius), center_x + long_radius), center_y + short_radius]
    top_pt = [random.uniform(center_x - long_radius, center_x + long_radius), center_y - short_radius]
    pts = np.array([lb_pt, rb_pt, top_pt], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, BLACK, 3)
    return img


def draw_rectangle(img, size):
    long_radius = sizable_radius(size)
    short_radius = int(long_radius * random.uniform(0.5, 0.8))
    center_x, center_y = safe_rand_center(long_radius, short_radius)
    cv2.rectangle(img, (center_x - long_radius, center_y - short_radius), (center_x + long_radius, center_y + short_radius), BLACK, 3)
    return img


def draw_square(img, size):
    radius = sizable_radius(size)
    center_x, center_y = safe_rand_center(radius, radius)
    cv2.rectangle(img, (center_x - radius, center_y - radius), (center_x + radius, center_y + radius), BLACK, 3)
    return img


shape_routes = {
    'Circle': draw_circle,
    'Ellipse': draw_ellipse,
    'Triangle': draw_triangle,
    'Rectangle': draw_rectangle,
    'Square': draw_square,
}


def distort(img, factor=0.1, prob_chg_dir=0.1):
    A = img.shape[0] / 4.0
    w = 2.0 / img.shape[1]
    shift = lambda x: A * np.sin(2.0 * np.pi * x * w) / (1 / factor)
    k, dir = 0, 1
    for i in range(img.shape[0]):
        if random.random() < prob_chg_dir:
            dir = -dir
        k += dir
        img[:, i] = np.roll(img[:,i], int(shift(k)))

    A = img.shape[1] / 4.0
    w = 2.0 / img.shape[0]
    shift = lambda x: A * np.sin(2.0 * np.pi * x * w) / (1 / factor)
    k, dir = 0, 1
    for i in range(img.shape[0]):
        if random.random() < prob_chg_dir:
            dir = -dir
        k += dir
        img[i, :] = np.roll(img[i,:], int(shift(k)))
    return img


def rotate(img, degree):
    rows, cols = img.shape[:2]
    R = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
    dst = cv2.warpAffine(img, R, (cols, rows), borderValue = WHITE)
    return dst


def draw_one(shape, size, factor, degree):
    img = create_img()
    img = shape_routes[shape](img, size)
    img = distort(img, factor)
    img = rotate(img, degree)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit(0)
    return img


def draw_many(shape, num_per_set, save_path):
    global count
    global rd
    for i in range(num_per_set):
        size = random.choice([0, 1])
        factor = random.uniform(0.1, 0.2)
        degree = rd.next()
        count += 1
        print('Drawing #{0} {1} with size = {2}, factor = {3}, degree = {4}'.format(count, shape, size, factor, degree))
        img = draw_one(shape, size, factor, degree)
        filename = '{0:04d}_{1}.png'.format(count, shape)
        filepath = os.path.join(save_path, filename)
        cv2.imwrite(filepath, img)


def generate_data(num_per_set, train_ratio=0.8, devel_ratio=0.1, test_ratio=0.1):
    global count
    num_train = int(num_per_set * train_ratio)
    num_devel = int(num_per_set * devel_ratio)
    num_test = num_per_set - num_train - num_devel

    for shape in shape_list:
        count = 0
        draw_many(shape, num_train, train_image_dir)
        draw_many(shape, num_devel, devel_image_dir)
        draw_many(shape, num_test, test_image_dir)


if __name__ == '__main__':
    random.seed(time.time())
    global rd
    rd = RandomDegree()
    generate_data(2)
