import cv2
import numpy as np
import math
import copy

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
        self.p1 = p1
        self.p2 = p2

    def __eq__(self, other):
        if self.p1 == other.p1 and self.p2 == other.p2:
            return True
        if self.p1 == other.p2 and self.p2 == other.p1:
            return True
        return False

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

    for i in range(len(list_of_points)):
        e = Edge(list_of_points[i-1], list_of_points[i])
        edges.append(e)

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
    img = cv2.pyrDown(cv2.imread("operator.jpg", cv2.IMREAD_UNCHANGED))
    out = np.zeros((img.shape[0], img.shape[1], 3))
    imgout = out

    for edge1 in list_of_edges:

        for edge2 in list_of_edges:
            if edge2.intersects(edge1) or edge2 == edge1:
                continue
            for edge3 in list_of_edges:
                if edge3.intersects(edge2) or edge3.intersects(edge1) or edge3 == edge1 or edge3 == edge2:
                    continue
                if can_make_tri(edge1, edge2, edge3):
                    #print((edge1.p1, edge1.p2), (edge2.p1, edge2.p2), (edge3.p1, edge3.p2))
                    t = Triangle(edge1, edge2, edge3)
                    to_add = True
                    for tri in triangles_list:
                        if t == tri:
                            to_add = False
                    if to_add:
                        triangles_list.append(t)
                        """
                        edge1.draw(imgout, (0, 0, 255))
                        edge2.draw(imgout, (0, 255, 0))
                        edge3.draw(imgout, (255, 0, 0))
                        cv2.imwrite("testimg.jpg", imgout)
                        """
                    
    return triangles_list

def find_color(tri_as_cont, image):
    x = int(((tri_as_cont[0, 0, 0] + tri_as_cont[1, 0, 0] + tri_as_cont[2, 0, 0])/3))
    y = int(((tri_as_cont[0, 0, 1] + tri_as_cont[1, 0, 1] + tri_as_cont[2, 0, 1])/3))
    color = (image[y][x][0], image[y][x][1], image[y][x][2])
    color = tuple([int(x) for x in color])
    return color

if __name__ == '__main__':
    img = cv2.pyrDown(cv2.imread("operator.jpg", cv2.IMREAD_UNCHANGED))
    out = np.zeros((img.shape[0], img.shape[1], 3))

    my_contour = [(10, 10), (50, 100), (10, 250), (250, 210), (500, 250), (300, 130), (500, 10), (400, 40), (300, 10), (200, 50), (100, 20), (50, 10), (33, 20)]
    """
    initpoint = contour[0]
    prevpoint = initpoint
    for point in contour[1:]:
        cv2.line(out, prevpoint, point, (255, 0, 255))
        prevpoint = point
    cv2.line(out, prevpoint, initpoint, (255, 0, 255))
    """
    """
    img = cv2.pyrDown(cv2.imread("operator.jpg", cv2.IMREAD_UNCHANGED))
    out = np.zeros((img.shape[0], img.shape[1], 3))

    h_thresh = int(img.shape[0] / 16)
    w_thresh = int(img.shape[1] / 32)
    print(img.shape[0], img.shape[1])
    

    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 127, 255, cv2.THRESH_BINARY_INV)
    contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours[:] = [c for c in contours if contour_threshold(c, h_thresh, w_thresh)]

    #contours is a python list of numpy 3d arrays (num_points, 1(redund), 2(x, y))
    shapes_coords = []

    #currently this loop does two things: 
        #first, it breaks complicated contours into approximations and then draws them
        #second, it takes those approximations and reconstructs them back into contours (in the shapes_contour list)
    for cont in contours:
        epsilon = (1/(h_thresh * w_thresh)) * 1 * cv2.arcLength(cont, True)
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
            #print("###", type(prev_point))
            #print("###", prev_point)
            cv2.line(out, prev_point, curr_point, color)
            prev_point = curr_point
            if i == len(approx) - 1:
                cv2.line(out, curr_point, init_point, (0, 0, 255))
            curr_shape_coords.append(curr_point)
        shapes_coords.append(curr_shape_coords)
    """

    triangles = triangulate(my_contour)
    #cv2.drawContours(out, triangles, -1, (255, 0, 0))
    print(type(triangles))
    print(type(triangles[0]))
    for t in triangles:
        print(t.shape)


    for i in range(len(triangles)):
        color = find_color(triangles[i], img)
        print(color)
        color = tuple([int(x) for x in color])
        cv2.drawContours(out, triangles, i, color, thickness=cv2.FILLED)


    cv2.imwrite("output2.jpg", out)

    