import cv2
import numpy as np
import math
import copy
import random


def contour_threshold(contour, h_thresh, w_thresh):
    x,y,w,h = cv2.boundingRect(contour)
    if w_thresh * h_thresh > w * h:
        return False
    return True

def side_of_line(n, p1, p2):
    x, y = n
    x1, y1 = p1
    x2, y2 = p2

    return (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)

def intersects_any(this_edge, list_of_edges):
    for edge in list_of_edges:
        if this_edge.intersects(edge):
            return True
    return False



class Edge:
    def __init__(self, p1, p2):
        self.p1 = p1 #these are tuples (x,y)
        self.p2 = p2

    def __eq__(self, other):
        if self.p1 == other.p1 and self.p2 == other.p2:
            return True
        if self.p1 == other.p2 and self.p2 == other.p1:
            return True
        return False

    def __hash__(self):
        return hash((self.p1, self.p2))

    def intersects(self, line_in):
        a = side_of_line(line_in.p1, self.p1, self.p2)
        b = side_of_line(line_in.p2, self.p1, self.p2)
        c = side_of_line(self.p1, line_in.p1, line_in.p2)
        d = side_of_line(self.p2, line_in.p1, line_in.p2)
        return True if (abs(a + b) < abs(a) + abs(b)) and (abs(c + d) < abs(c) + abs(d)) else False

    def draw(self, canvas, color=(0, 255, 255)):
        cv2.line(canvas, self.p1, self.p2, color)

    def connected(self, other):
        if self == other:
            return False
        if self.p1 == other.p1 or self.p1 == other.p2 or self.p2 == other.p1 or self.p2 == other.p2:
            return True
        return False

def connected(e1, e2, e3):
    vertexset = {e1.p1, e1.p2, e2.p1, e2.p2, e3.p1, e3.p2}
    length = len(vertexset)
    if length == 3:
        return True
    return False
    
def triangulate(list_of_points):
    edges = []

    #draws original contour
    for i in range(len(list_of_points)):
        e = Edge(list_of_points[i-1], list_of_points[i])
        edges.append(e)

    lop_a = copy.deepcopy(list_of_points)
    random.shuffle(lop_a)
    lop_b = copy.deepcopy(list_of_points)
    random.shuffle(lop_b)

    for point1 in list_of_points:
        for point2 in list_of_points:
            if point1 == point2:
                continue
            curr_edge = Edge(point1, point2)
            if not intersects_any(curr_edge, edges):
                edges.append(curr_edge)

    triangles = break_into_triangles(edges)
    contours_list = []
    for t in triangles:
        a = t.transform_into_contour()
        contours_list.append(a)
    return contours_list

class Triangle:
    def __init__(self, e1, e2, e3):
        self.p1 = e1.p1
        self.p2 = e1.p2
        if e2.p1 != self.p1 and e2.p1 != self.p2:
            self.p3 = e2.p1
        else:
            self.p3 = e2.p2

    def __eq__(self, tri):
        d1 = {self.p1, self.p2, self.p3}
        d2 = {tri.p1, tri.p2, tri.p3}
        return d1 == d2

    def draw(self, canvas, color=(0, 255, 255)):
        cv2.line(canvas, self.p1, self.p2, color)
        cv2.line(canvas, self.p1, self.p3, color)
        cv2.line(canvas, self.p3, self.p2, color)

    def transform_into_contour(self):
        #contours is a python list of numpy 3d arrays (num_points, 1(redund), 2(x, y))
        contour = np.zeros((3, 1, 2))
        contour[0, 0, 0] = self.p1[0]
        contour[0, 0, 1] = self.p1[1]
        contour[1, 0, 0] = self.p2[0]
        contour[1, 0, 1] = self.p2[1]
        contour[2, 0, 0] = self.p3[0]
        contour[2, 0, 1] = self.p3[1]
        return contour.astype(int)

    def find_color(self, image):
        x = int((self.p1[0] + self.p2[0] + self.p3[0])/3)
        y = int((self.p1[1] + self.p2[1] + self.p3[1])/3)
        return (image[y][x][0], image[y][x][1], image[y][x][2])

def can_make_tri(e1, e2, e3):
    vertexset = {e1.p1, e1.p2, e2.p1, e2.p2, e3.p1, e3.p2}
    no_intersections = (not e1.intersects(e2) and not e2.intersects(e3) and not e3.intersects(e1))
    if len(vertexset) == 3 and no_intersections:
        return True
    return False

def break_into_triangles(list_of_edges):
    triangles_list = []

    my_edges = {}
    for edge in list_of_edges:
        my_edges[edge] = []
    
    for curr_edge in list_of_edges:
        for maybe_edge in list_of_edges:
            if curr_edge.connected(maybe_edge):
                my_edges[curr_edge].append(maybe_edge)

    for first_edge in list_of_edges:
        for second_edge in my_edges[first_edge]:
            for third_edge in my_edges[second_edge]:
                if can_make_tri(first_edge, second_edge, third_edge):
                    t = Triangle(first_edge, second_edge, third_edge)
                    to_add = True
                    for tri in triangles_list:
                        if t == tri:
                            to_add = False
                            break
                    if to_add:
                        triangles_list.append(t)
    return triangles_list

def find_color(tri_as_cont, image):
    x = int(((tri_as_cont[0, 0, 0] + tri_as_cont[1, 0, 0] + tri_as_cont[2, 0, 0])/3))
    y = int(((tri_as_cont[0, 0, 1] + tri_as_cont[1, 0, 1] + tri_as_cont[2, 0, 1])/3))
    color = (image[y][x][0], image[y][x][1], image[y][x][2])
    color = tuple([int(x) for x in color])
    return color

def approx_contour_area(contour):
    x,y,w,h = cv2.boundingRect(contour)
    return (w * h)



if __name__ == '__main__':
    img = cv2.pyrDown(cv2.imread("eotech.jpg", cv2.IMREAD_UNCHANGED))
    out = np.zeros((img.shape[0], img.shape[1], 3))

    h_thresh = int(img.shape[0] / 16)
    w_thresh = int(img.shape[1] / 32)
    print(img.shape[0], img.shape[1])
    

    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 127, 255, cv2.THRESH_BINARY_INV)
    contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours[:] = [c for c in contours if contour_threshold(c, h_thresh, w_thresh)]

    shapes_coords = []

    contours = sorted(contours, key=approx_contour_area, reverse=True)
    
    for cont in contours:
        if approx_contour_area(cont) < ((h_thresh * w_thresh)):
            continue
        epsilon = .01 * cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, epsilon, True)
        init_point = (approx[0,0,0], approx[0,0,1])
        curr_shape_coords = [init_point]
        prev_point = init_point
        for i in range(1, len(approx)):
            if i == 1:
                color = (0, 255, 0)
            else:
                color = (255, 255, 255)
            curr_point = (approx[i,0,0], approx[i,0,1])
            cv2.line(out, prev_point, curr_point, color)
            prev_point = curr_point
            if i == len(approx) - 1:
                cv2.line(out, curr_point, init_point, (0, 0, 255))
            curr_shape_coords.append(curr_point)
        shapes_coords.append(curr_shape_coords)


    triangles_list = []
    for shape in shapes_coords:
        triangles_list.append(triangulate(shape))

    for triangles in triangles_list:
        for i in range(len(triangles)):
            color = find_color(triangles[i], img)
            print(color)
            color = tuple([int(x) for x in color])
            cv2.drawContours(out, triangles, i, color, thickness=cv2.FILLED)


    cv2.imwrite("output2.jpg", out)

    