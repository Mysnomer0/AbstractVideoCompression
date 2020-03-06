import cv2
import numpy as np
import math
import random

def GetAveragePixelValueInsideRectangle(image, xOfRectangle, yOfRectangle, widthOfRectangle, heightOfRectangle):
    return (np.mean(image[yOfRectangle:yOfRectangle + heightOfRectangle, xOfRectangle:xOfRectangle + widthOfRectangle, 0]), \
        np.mean(image[yOfRectangle:yOfRectangle + heightOfRectangle, xOfRectangle:xOfRectangle + widthOfRectangle, 1]), \
            np.mean(image[yOfRectangle:yOfRectangle + heightOfRectangle, xOfRectangle:xOfRectangle + widthOfRectangle, 2]))

def contour_threshold(contour, h_thresh, w_thresh):
    x,y,w,h = cv2.boundingRect(contour)
    if w_thresh * h_thresh > w * h:
        return False
    return True

def euc_dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def side_of_line(n, p1, p2):
    x, y = n
    x1, y1 = p1
    x2, y2 = p2

    return (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)

class Polygon:
    #any number of triangles
    def __init__(self, triangle=None):
        if triangle is None:
            self.triangles = []
        else:
            self.triangles = [triangle]

class Triangle:
    #p will be a contour containing three points
    def __init__(self, arg):
        self.p = []

class LineSegment:
    def __init__(self, pa, pb):
        self.points = [(pa[0], pa[1]), (pb[0], pb[1])]

    def intersects(self, line_in):
        fst = side_of_line(line_in.points[0], self.points[0], self.points[1])
        sec = side_of_line(line_in.points[1], self.points[0], self.points[1])
        return True if abs(fst + sec) < abs(fst) + abs(sec) else False
    
    def contains(self, point):
        return True if point in self.points else False

    def draw(self, canvas):
        cv2.line(canvas, self.points[0], self.points[1], (255, 0, 255))

def is_neighbor_of(p1, p2, list_of_points):
    for i in range(len(list_of_points)):
        if list_of_points[i] == p1:
            if i == len(list_of_points) - 1: #special case
                if list_of_points[i-1] == p2 or list_of_points[-1] == p2:
                    return True
            else:
                if list_of_points[i-1] == p2 or list_of_points[i+1] == p2:
                    return True
        if list_of_points[i] == p2:
            if i == len(list_of_points) - 1:
                if list_of_points[i-1] == p1 or list_of_points[-1] == p1:
                    return True
            else:
                if list_of_points[i-1] == p1 or list_of_points[i+1] == p1:
                    return True

    return False

def triangulate(list_of_points):

    #milan method:
        #connect our original shape with lines (in-order in the list)
        #connect every point with lines (remembering our origin)
        #delete any lines that are in the convex hull
        #delete any lines that are outside the shape (STRETCH GOAL)
        #for each line:
            #for each other line:
                #if they overlap, delete one of them at random.
        #triangulated polygon!
    hull = []
    init_point = list_of_points[0]
    prev_point = init_point
    for point in list_of_points[1:]:
        a = LineSegment(prev_point, point)
        hull.append(a)
        prev_point = point
    a = LineSegment(prev_point, init_point)
    hull.append(a)
    return hull

def generate_random_hull(x, y, count):
    list_of_points = []
    for i in range(count):
        list_of_points.append((random.randrange(0, x), random.randrange(0, y)))
    return list_of_points



            
if __name__ == '__main__':
    img = cv2.pyrDown(cv2.imread("operator.jpg", cv2.IMREAD_UNCHANGED))
    out = np.zeros((img.shape[0], img.shape[1], 3))
    print(out.shape)

    list_of_points = generate_random_hull(img.shape[1], img.shape[0], 20)
    hull = triangulate(list_of_points)
    for line in hull:
        line.draw(out)


    cv2.imwrite("output.jpg", out)

        