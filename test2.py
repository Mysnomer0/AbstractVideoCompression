import cv2
import numpy as np
import math
import copy
import random

#used to determine whether or not a contour is large enough to bother drawing
def contour_threshold(contour, h_thresh, w_thresh):
    x,y,w,h = cv2.boundingRect(contour)
    if w_thresh * h_thresh > w * h:
        return False
    return True

def approx_contour_area(contour):
    x,y,w,h = cv2.boundingRect(contour)
    return (w * h)

#helper function to see if the edge we want to draw will intersect any
#   existing edges in a list of edges
def intersects_any(this_edge, list_of_edges):
    for edge in list_of_edges:
        if this_edge.intersects(edge):
            return True
    return False

#given 3 edges, can we connect those edges into a triangle or not?
def can_make_tri(e1, e2, e3):
    vertexset = {e1.p1, e1.p2, e2.p1, e2.p2, e3.p1, e3.p2}
    no_intersections = (not e1.intersects(e2) and not e2.intersects(e3) and not e3.intersects(e1))
    if len(vertexset) == 3 and no_intersections:
        return True
    return False

#find the midpoint of a triangle, and get the color of the original image at that point
def find_color(tri_as_cont, image):
    x = int(((tri_as_cont[0, 0, 0] + tri_as_cont[1, 0, 0] + tri_as_cont[2, 0, 0])/3))
    y = int(((tri_as_cont[0, 0, 1] + tri_as_cont[1, 0, 1] + tri_as_cont[2, 0, 1])/3))
    color = (image[y][x][0], image[y][x][1], image[y][x][2])
    color = tuple([int(x) for x in color])
    return color


class Edge:
    def __init__(self, p1, p2):
        self.p1 = p1 #these are tuples (x,y)
        self.p2 = p2

    #will return true for antiparallel edges
    def __eq__(self, other):
        if self.p1 == other.p1 and self.p2 == other.p2:
            return True
        if self.p1 == other.p2 and self.p2 == other.p1:
            return True
        return False

    #need this to use dictionaries/sets
    def __hash__(self):
        return hash((self.p1, self.p2))

    #does self intersect line_in?
    def intersects(self, line_in):
        def side_of_line(n, p1, p2):
            x, y = n
            x1, y1 = p1
            x2, y2 = p2
            return (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)
        a = side_of_line(line_in.p1, self.p1, self.p2)
        b = side_of_line(line_in.p2, self.p1, self.p2)
        c = side_of_line(self.p1, line_in.p1, line_in.p2)
        d = side_of_line(self.p2, line_in.p1, line_in.p2)
        return True if (abs(a + b) < abs(a) + abs(b)) and (abs(c + d) < abs(c) + abs(d)) else False

    #not currently used - draws cv2 lines for debugging
    def draw(self, canvas, color=(0, 255, 255)):
        cv2.line(canvas, self.p1, self.p2, color)

    #do self and other share an endpoint?
    def connected(self, other):
        if self == other:
            return False
        if self.p1 == other.p1 or self.p1 == other.p2 or self.p2 == other.p1 or self.p2 == other.p2:
            return True
        return False

class Triangle:
    #given 3 edges, draw a triangle. No guards here as they are in other functions (can_make_tri())
    def __init__(self, e1, e2, e3):
        self.p1 = e1.p1
        self.p2 = e1.p2
        if e2.p1 != self.p1 and e2.p1 != self.p2:
            self.p3 = e2.p1
        else:
            self.p3 = e2.p2

    #will return true for the same triangle with a different ordering of points. 
    def __eq__(self, tri):
        d1 = {self.p1, self.p2, self.p3}
        d2 = {tri.p1, tri.p2, tri.p3}
        return d1 == d2

    #not currently used  - draws a triangle with cv2 lines for debugging
    def draw(self, canvas, color=(0, 255, 255)):
        cv2.line(canvas, self.p1, self.p2, color)
        cv2.line(canvas, self.p1, self.p3, color)
        cv2.line(canvas, self.p3, self.p2, color)

    #massages the triangle into the format expected by cv2.drawContours
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

#given a list of points - form [(x, y), (x, y), ...]
#   return a list of cv2 contours describing triangles that cover the same shape.
#   note that this function expects an input that has already been formatted with
#   cv2.approxPolyDP and converted from cv2 3d array contours into the format above. 
def triangulate(list_of_points):
    edges = []

    #make sure our original contour is always in edges
    for i in range(len(list_of_points)):
        e = Edge(list_of_points[i-1], list_of_points[i])
        edges.append(e)

    #shuffle original list so we don't get all triangles sharing a vertex (list_of_points[0])
    lop_a = copy.deepcopy(list_of_points)
    random.shuffle(lop_a)
    lop_b = copy.deepcopy(list_of_points)
    random.shuffle(lop_b)

    #find each possible line, check if it intersects anything already in the edge
    #   set - if not, add to the edge set. 
    for point1 in lop_a:
        for point2 in lop_b:
            if point1 == point2:
                continue
            curr_edge = Edge(point1, point2)
            if not intersects_any(curr_edge, edges):
                edges.append(curr_edge)

    #call helper function
    triangles = break_into_triangles(edges)
    contours_list = []
    
    #massage output so opencv can use it.
    for t in triangles:
        a = t.transform_into_contour()
        contours_list.append(a)
    return contours_list

#helper function for triangulate - given a list of Edges, split that list of edges 
#   into a list of opencv contours describing individual triangles. 
def break_into_triangles(list_of_edges):
    triangles_list = []

    #initialize dict so each edge is a key to an empty list
    my_edges = {}
    for edge in list_of_edges:
        my_edges[edge] = []
    
    #for each edge, add to its list every edge that is connected to it. 
    #   note that connected returns false when the edges are parallel or antiparallel
    for curr_edge in list_of_edges:
        for maybe_edge in list_of_edges:
            if curr_edge.connected(maybe_edge):
                my_edges[curr_edge].append(maybe_edge)

    #for each edge, grab all pairs of connected edges and check if they form a triangle.
    #   if so, add the triangle to triangles_list. 
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

def preprocess_image(image):
    return cv2.GaussianBlur(image, (13,13), 0)


if __name__ == '__main__':

    # Read in our image, init our output image. 
    img = cv2.pyrDown(cv2.imread("hotel2.jpg", cv2.IMREAD_UNCHANGED))
    # Define our output image
    out = np.zeros((img.shape[0], img.shape[1], 3))

    # Preprocess the image
    img = preprocess_image(img)

    #cv2.imwrite("output.jpg", img)
    
    # Thresholds for shape size
    h_thresh = int(img.shape[0] / 16)
    w_thresh = int(img.shape[1] / 32)
    print(img.shape[0], img.shape[1])
    
    #find the cv2 contours of the image
    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 127, 255, cv2.THRESH_BINARY_INV)
    contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #throw away some contours
    contours[:] = [c for c in contours if contour_threshold(c, h_thresh, w_thresh)]
    shapes_coords = []
    #sort the contours in order of descending area - so background contours appear behind foreground contours. 
    contours = sorted(contours, key=approx_contour_area, reverse=True)
    for cont in contours:
        #throw away some more contours. 
        if approx_contour_area(cont) < ((h_thresh * w_thresh)):
            continue
        #epsilon is the degree to which our approximation remains close to the original contour
        #smaller epsilon means polygons with more vertices. 
        epsilon = .01 * cv2.arcLength(cont, True)
        #find an approximation for current contour
        approx = cv2.approxPolyDP(cont, epsilon, True)
        init_point = (approx[0,0,0], approx[0,0,1])
        curr_shape_coords = [init_point]
        prev_point = init_point
        #this loop builds a more sane data structure for contours, transforming a
        #   list of 3d numpy arrays into a list of (x,y) tuples. The cv2.line bit
        #   is to assist with debugging. 
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
            #curr_shape_coords is the list of coordinate points for the approximation of the current contour.
            curr_shape_coords.append(curr_point)
        #shapes_coords is the list of lists of coordinate points for the approximation of the current contour. 
        shapes_coords.append(curr_shape_coords)

    #triangles_list has form: list of lists of numpy 3d numpy arrays 
    triangles_list = []
    for shape in shapes_coords:
        triangles_list.append(triangulate(shape))

    #finally, for each list, for each triangle in that list, find its color and draw the triangle. 
    for triangles in triangles_list:
        for i in range(len(triangles)):
            color = find_color(triangles[i], img)
            print(color)
            color = tuple([int(x) for x in color])
            cv2.drawContours(out, triangles, i, color, thickness=cv2.FILLED)

    #write output, and we're done. 
    cv2.imwrite("output.jpg", out)
    
