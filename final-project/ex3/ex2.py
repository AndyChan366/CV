#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# disjoint-set
class universe():
    def __init__(self, elements):
        self.num = elements
        self.elts = []
        for i in range(elements):
            rank = 0
            size = 1
            p = i
            self.elts.append((rank, size, p))

    def find(self, u):
        if self.elts[u][2] == u:
            return u
        self.elts[u] = (self.elts[u][0], self.elts[u][1], self.find(self.elts[u][2]))
        return self.elts[u][2]

    def join(self, x, y):
        if self.elts[x][0] > self.elts[y][0]:
            self.elts[y] = (self.elts[y][0], self.elts[y][1], self.elts[x][2])
            self.elts[x] = (self.elts[x][0], self.elts[x][1] + self.elts[y][1], self.elts[x][2])
        else:
            self.elts[x] = (self.elts[x][0], self.elts[x][1], self.elts[y][2])
            self.elts[y] = (self.elts[y][0], self.elts[y][1] + self.elts[x][1], self.elts[y][2])
            if self.elts[x][0] == self.elts[y][0]:
                self.elts[y] = (self.elts[y][0] + 1, self.elts[y][1], self.elts[y][2])
        self.num -= 1

    def size(self, x):
        return self.elts[x][1]

    def num_sets(self):
        return self.num
    
    def comps(self):
        ret = []
        i, n = 0, 0
        while n != self.num:
            if self.elts[i][2] == i:
                ret.append(i)
                n += 1
            i += 1
        return ret


# In[3]:


def criteria(img, x1, y1, x2, y2):
    r = np.square(img[0][y1, x1] - img[0][y2, x2])
    g = np.square(img[1][y1, x1] - img[1][y2, x2])
    b = np.square(img[2][y1, x1] - img[2][y2, x2])
    return np.sqrt(r + g + b)

def THRESHOLD(size, c):
    return c / size


# In[4]:


# Segment a graph
# Returns a disjoint-set forest representing the segmentation.

def segment_graph(num_vertices, num_edges, graph, c):
    # make a disjoint-set forest
    u = universe(num_vertices)
    # init thresholds
    threshold = np.zeros(num_vertices, dtype=float)
    for i in range(num_vertices):
        threshold[i] = THRESHOLD(1, c)
    # for each edge, in non-decreasing weight orders
    for i in range(num_edges):
        a = u.find(graph[i][0])
        b = u.find(graph[i][1])
        if a != b:
            if (graph[i][2] <= threshold[a]) and graph[i][2] <= threshold[b]:
                u.join(a, b)
                a = u.find(a)
                threshold[a] = graph[i][2] + THRESHOLD(u.size(a), c)
    return u


# In[5]:


# Segment an image
# This function is used to solve homework ex2
# return: gt(after mark)
def segment_image(im, sigma, c, min_size, num_ccs, gt):
    height, width, channel = im.shape
    im = np.array(im, dtype=float)
    gaussian_img = cv2.GaussianBlur(im, (5, 5), sigma)
    b, g, r = cv2.split(gaussian_img)
    smooth_img = (r, g, b)
    graph = []
    num = 0
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                a = y * width + x
                b = y * width + (x + 1)
                w = criteria(smooth_img, x, y, x + 1, y)
                num += 1
                graph.append((a, b, w))
            if y < height - 1:
                a = y * width + x
                b = (y + 1) * width + x
                w = criteria(smooth_img, x, y, x, y + 1)
                num += 1
                graph.append((a, b, w))
            if x < width - 1 and y < height - 1:
                a = y * width + x
                b = (y + 1) * width + (x + 1)
                w = criteria(smooth_img, x, y, x + 1, y + 1)
                num += 1
                graph.append((a, b, w))
            if x < width - 1 and y > 0:
                a = y * width + x
                b = (y - 1) * width + (x + 1)
                w = criteria(smooth_img, x, y, x + 1, y - 1)
                num += 1
                graph.append((a, b, w))
    # sort according to the similarity
    graph = sorted(graph, key=lambda x: (x[2]))
    u = segment_graph(width * height, num, graph, c)
    for i in range(num):
        a = u.find(graph[i][0])
        b = u.find(graph[i][1])
        # if the number of pixel in each area < 50, concat.
        if (a != b) and ((u.size(a) < min_size) or u.size(b) < min_size):
            u.join(a, b)
    # dynamic adjust to keep the number of area is [50,70]
    while u.num_sets() < 50 or u.num_sets() > 70:
        if u.num_sets() < 50:
            c = c / 2
            u = segment_graph(width * height, num, graph, c)
        if u.num_sets() > 70:
            c = c * 1.5
            u = segment_graph(width * height, num, graph, c)
        for i in range(num):
            a = u.find(graph[i][0])
            b = u.find(graph[i][1])
            if (a != b) and ((u.size(a) < min_size) or u.size(b) < min_size):
                u.join(a, b)
    num_ccs.append(u.num_sets())
    comps = u.comps()
    # calculate the number of white pixel(gt) in each area
    cnt = [0 for i in range(u.num_sets())]
    # used to mark the segmentation area
    mark = [0 for i in range(u.num_sets())]
    for y in range(height):
        for x in range(width):
            comp = u.find(y * width + x)
            # Here I use data/gt as reference, if the current pixel in data/gt is gt, +1
            if (gt[y, x, :] == [255, 255, 255]).all():
                cnt[comps.index(comp)] += 1
    # if the white pixel is more than half, then mark the area into gt.
    for i in range(u.num_sets()):
        if 2 * cnt[i] >= u.size(comps[i]):
            mark[i] = 1
        else:
            mark[i] = 0
    # final gt after mark
    for y in range(height):
        for x in range(width):
            comp = u.find(y * width + x)
            if mark[comps.index(comp)] == 1:
                gaussian_img[y, x, :] = [255, 255, 255]
            else:
                gaussian_img[y, x, :] = [0, 0, 0]

    return gaussian_img


# In[6]:


# Segment an image
# This function is only for ex3 in homework
# return: gt(after mark), every little area(segmentation), the count of area, label(0:bg, 1:gt)
def segment_image_forthree(im, sigma, c, min_size, num_ccs, gt):
    height, width, channel = im.shape
    im = np.array(im, dtype=float)
    gaussian_img = cv2.GaussianBlur(im, (5, 5), sigma)
    b, g, r = cv2.split(gaussian_img)
    smooth_img = (r, g, b)
    graph = []
    num = 0
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                a = y * width + x
                b = y * width + (x + 1)
                w = criteria(smooth_img, x, y, x + 1, y)
                num += 1
                graph.append((a, b, w))
            if y < height - 1:
                a = y * width + x
                b = (y + 1) * width + x
                w = criteria(smooth_img, x, y, x, y + 1)
                num += 1
                graph.append((a, b, w))
            if x < width - 1 and y < height - 1:
                a = y * width + x
                b = (y + 1) * width + (x + 1)
                w = criteria(smooth_img, x, y, x + 1, y + 1)
                num += 1
                graph.append((a, b, w))
            if x < width - 1 and y > 0:
                a = y * width + x
                b = (y - 1) * width + (x + 1)
                w = criteria(smooth_img, x, y, x + 1, y - 1)
                num += 1
                graph.append((a, b, w))
     # sort according to the similarity
    graph = sorted(graph, key=lambda x: (x[2]))
    u = segment_graph(width * height, num, graph, c)
    for i in range(num):
        a = u.find(graph[i][0])
        b = u.find(graph[i][1])
        # if the number of pixel in each area < 50, concat.
        if (a != b) and ((u.size(a) < min_size) or u.size(b) < min_size):
            u.join(a, b)
    # dynamic adjust to keep the number of area is [50,70]
    while u.num_sets() < 50 or u.num_sets() > 70:
        if u.num_sets() < 50:
            c = c / 2
            u = segment_graph(width * height, num, graph, c)
        if u.num_sets() > 70:
            c = c * 1.5
            u = segment_graph(width * height, num, graph, c)
        for i in range(num):
            a = u.find(graph[i][0])
            b = u.find(graph[i][1])
            if (a != b) and ((u.size(a) < min_size) or u.size(b) < min_size):
                u.join(a, b)
    num_ccs.append(u.num_sets())
    comps = u.comps()
    # calculate the number of white pixel(gt) in each area
    cnt = [0 for i in range(u.num_sets())]
    # used to mark the segmentation area
    mark = [0 for i in range(u.num_sets())]
    for y in range(height):
        for x in range(width):
            comp = u.find(y * width + x)
            # Here I use data/gt as reference, if the current pixel in data/gt is gt, +1
            if (gt[y, x, :] == [255, 255, 255]).all():
                cnt[comps.index(comp)] += 1
    # if the white pixel is more than half, then mark the area into gt.
    for i in range(u.num_sets()):
        if 2 * cnt[i] >= u.size(comps[i]):
            mark[i] = 1
        else:
            mark[i] = 0
    # from here, I add some statements to finish homework ex3.
    block = {}       # each area
    label = {}       # each area's mark(gt/bg)
    # get the histogram of each little area after segmentation
    for y in range(height):
        for x in range(width):
            comp = u.find(y * width + x)
            if comp not in block:
                block[comp] = np.zeros((8,8,8))
                block[comp][int(smooth_img[0][y][x] // 32)][int(smooth_img[1][y][x] // 32)][int(smooth_img[2][y][x] // 32)] += 1
            else:
                block[comp][int(smooth_img[0][y][x] // 32)][int(smooth_img[1][y][x] // 32)][int(smooth_img[2][y][x] // 32)] += 1
    # flatten:(8 * 8 * 8) -> (1 * 512)
    for comp in block:
        block[comp] = block[comp].flatten()
        sum = np.sum(block[comp])
        for i in range(len(block[comp])):
            block[comp][i] = block[comp][i] / sum
        # print(block[comp].shape)
    # print(len(block))
    # final gt after mark
    for y in range(height):
        for x in range(width):
            comp = u.find(y * width + x)
            if mark[comps.index(comp)] == 1:
                gaussian_img[y, x, :] = [255, 255, 255]
                label[comp] = 1
            else:
                gaussian_img[y, x, :] = [0, 0, 0]
                label[comp] = 0

    return gaussian_img, block, u.num_sets(), label


# In[7]:


def IOU(resseg, gt):
    height, width, channel = resseg.shape
    I, U = 0, 0
    for y in range(height):
        for x in range(width):
            if (gt[y, x, :] == [255, 255, 255]).all() and (resseg[y, x, :] == [255, 255, 255]).all():
                I += 1
            if (gt[y, x, :] == [255, 255, 255]).all() or (resseg[y, x, :] == [255, 255, 255]).all():
                U += 1
    return float(I) / float(U)


# In[9]:


# sigma = 0.8
# k = 60
# min_size = 50
print("Please wait...")
for i in range(10):
    pic = "../data/imgs/{}.png".format(i * 100 + 13)
    ground = "../data/gt/{}.png".format(i * 100 + 13)
    input = cv2.imread(pic)
    gt = cv2.imread(ground)
    num_ccs = []
    res = segment_image(input, 0.8, 60, 50, num_ccs, gt)
    cv2.imwrite("resseg/{}.png".format(i * 100 + 13), res)
    print(IOU(res, gt))


# In[ ]:




