import cv2
import numpy as np
import math

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

"""    
def triangulate(list_of_points):
    #we can assume that the list is sorted - ie that sequential points are neighbors.
    #the trick is to preserve that ordering! (i think.)

    if(len(list_of_points) == 3):
        return [(list_of_points[0], list_of_points[1], list_of_points[2])]

    largest_dist = 0
    point1 = None
    point2 = None

    for start_point in list_of_points:
        for next_point in list_of_points:
            if is_neighbor_of(start_point, next_point, list_of_points):
                continue
            curr_dist = euc_dist(start_point, next_point)
            if curr_dist > largest_dist:
                largest_dist = curr_dist
                point1 = start_point
                point2 = next_point

    #Now we have a cycle of points, and a pair of points that we will use to define our new two cycles.

    cycle_neg = [point1, point2]
    cycle_pos = [point1, point2]

    for point in list_of_points:
        if point == point1 or point == point2:
            continue
        if is_neighbor_of(cycle_neg[-1], point, list_of_points):
            if side < 0: 
                cycle_neg.append(point)



    flag = False
    point_to_remove = None
    cycle_neg = [point1, point2]
    cycle_pos = [point1, point2]
    for point in list_of_points:
        if flag == True:
            list_of_points.remove(point)
            flag = False
        if point == point1 or point == point2:
            continue
        flag = False
        side = side_of_line(point, point1, point2)
        if is_neighbor_of(cycle_neg[-1], point, list_of_points):
            #point can go in cycle 1
            if side < 0:
                cycle_neg.append(point)
                flag = True
                point_to_remove = point
                continue
        if is_neighbor_of(cycle_pos[-1], point, list_of_points):
            if side > 0:
                cycle_pos.append(point)
                flag = True
                point_to_remove = point
                continue

    return triangulate(cycle_neg) + triangulate(cycle_pos)




def split_into_triangles(contour_list):
    #arg is a list of points (as tuples) - we're trying to split this list into a series of triangles
    #first I'll try the naieve approach - every 3 points become a triangle
    #second thought - maybe its easier to do the recursive case first, with list concatenation

    # 1) find the most distant two points in the list (that are not neighbors)
    # 2) connect these points - now we have two cycles (the to joined points will appear in both cycles)
    # 3) at any point, if we have a list that only has 3 points in it - return it as a triangle

    # 1) i'll do this using euclidian distance although it is probably woefully inefficient

    triangles = []

    if(len(contour_list) == 3):
        triangles.append(contour_list)
        return triangles

    largest_dist = 0
    point1 = None
    point2 = None


    for start_point in contour_list:
        for next_point in contour_list:
            if start_point == next_point:
                continue
            curr_dist = euc_dist(start_point, next_point)
            if curr_dist > largest_dist:
                largest_dist = curr_dist
                next_largest_dist = largest_dist
                point1 = start_point
                point2 = next_point

    #   now we have our two points that we'll split the cycle on. 
    #   Next we need to actually split the cycle in two
    #   first we add both points to both cycles
    cycle_neg = [point1, point2]
    cycle_pos = [point1, point2]

    for point in contour_list:
        side = side_of_line(point, point1, point2)
        if side < 0:
            cycle_neg.append(point)
        elif side > 0:
            cycle_pos.append(point)
"""

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

        


        """
            points = [
                
            ]

            POINT1
            POINT2
            POINT3
        """


    #we can assume that the list is sorted - ie that sequential points are neighbors.
    #the trick is to preserve that ordering! (i think.)

    if(len(list_of_points) == 3):
        return [(list_of_points[0], list_of_points[1], list_of_points[2])]

    largest_dist = 0
    point1 = None
    point2 = None

    for start_point in list_of_points:
        for next_point in list_of_points:
            #if is_neighbor_of(start_point, next_point, list_of_points):
            #    continue
            curr_dist = euc_dist(start_point, next_point)
            if curr_dist > largest_dist:
                largest_dist = curr_dist
                point1 = start_point
                point2 = next_point

    #Now we have a cycle of points, the pair of points who are the farthest away from one another. 
    #we can use this pair of points to find a triangle.

    points_above = [point1, point2]
    points_below = [point1, point2]

    for point in list_of_points:
        if point != point1 and point != point2:
            side = side_of_line(point, point1, point2)
            if side > 0:
                points_above.append(point)
            if side < 0:
                points_below.append(point)
            # i THINK that this will preserve ordering.
            #because points are added to their respective lists 

    return triangulate(points_above) + triangulate(points_below)















if __name__ == '__main__':
    img = cv2.pyrDown(cv2.imread("operator.jpg", cv2.IMREAD_UNCHANGED))
    out = np.zeros((img.shape[0], img.shape[1], 3))

    h_thresh = int(img.shape[0] / 16)
    w_thresh = int(img.shape[1] / 32)
    print(img.shape[0], img.shape[1])
    

    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 127, 255, cv2.THRESH_BINARY_INV)
    contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours[:] = [c for c in contours if contour_threshold(c, h_thresh, w_thresh)]

    #contours is a python list of numpy 3d arrays (num_points, 1(redund), 2(x, y))
    shapes_contour = []

    shapes_coords = []


    #currently this loop does two things: 
        #first, it breaks complicated contours into approximations and then draws them
        #second, it takes those approximations and reconstructs them back into contours (in the shapes_contour list)
    for cont in contours:
        epsilon = (1/(h_thresh * w_thresh)) * 1 * cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, epsilon, True)
        init_point = (approx[0,0,0], approx[0,0,1])
        curr_shape_contour = np.zeros((len(approx), 1, 2), dtype=np.intc)
        curr_shape_contour[0,0,0] = init_point[0]
        curr_shape_contour[0,0,1] = init_point[1]
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
            curr_shape_contour[i,0,0] = curr_point[0]
            curr_shape_contour[i,0,1] = curr_point[1]
            curr_shape_coords.append(curr_point)
        shapes_contour.append(curr_shape_contour)
        shapes_coords.append(curr_shape_coords)

    #now lets try to break shapes_contour into triangles

    """"
    list_of_shapes = []

    for i in shapes_coords:
        curr_shape = triangulate(i)
        list_of_shapes.append(curr_shape)
    
    for tlist in list_of_shapes:
        for t in tlist:
            print(t)
        print("\n\n")
    """

    #cv2.drawContours(out, shapes_contour, -1, (255, 0, 0))
    cv2.imwrite("output.jpg", out)

        