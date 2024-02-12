import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime


# Read points from text file
def read_points(points):
    # Create an array of points.
    points_1 = []

    # Read points
    for key in points:
        for point in points[key]:
            points_1.append(point)
    # with open(path) as file:
    #     for line in file:
    #         x, y = line.split()
    #         points.append((int(x), int(y)))
    return points_1


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def apply_affine_transform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Check if a point is inside a rectangle
def rect_contains(rect, point):
    return not point[0] < rect[0] \
           or point[1] < rect[1] \
           or point[0] > rect[0] + rect[2] \
           or point[1] > rect[1] + rect[3]


# calculate delanauy triangle
def calculate_delaunay_triangles(rect, points):
    # create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    triangle_list = subdiv.getTriangleList()

    delaunay_tri = []

    pt = []

    for t in triangle_list:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        pt += [pt1, pt2, pt3]

        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            ind = []
            # Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0:
                        ind.append(k)
                        # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
            if len(ind) == 3:
                delaunay_tri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunay_tri


# Warps and alpha blends triangular regions from img1 and img2 to img
def warp_triangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (r2[2], r2[3])

    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

    img2_rect = img2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
            (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect


def start_swapping(path, file_name1, file_name2):
    # # Make sure OpenCV is version 3.0 or above
    # (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    #
    # if int(major_ver) < 3:
    #     print >> sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher'
    #     sys.exit(1)

    # Read images
    file_name1 = os.path.join(path, file_name1)
    file_name2 = os.path.join(path, file_name2)

    # img1d = dlib.load_rgb_image(file_name1)
    # img2d = dlib.load_rgb_image(file_name2)


    # image1 = face_recognition.load_image_file(filename1)
    # image2 = face_recognition.load_image_file(filename2)

    # Read array of corresponding points

    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    #
    # rect1 = detector(img1d)[0]
    # sp1 = predictor(img1d, rect1)
    # points1 = [(p.x, p.y) for p in sp1.parts()]
    #
    # rect2 = detector(img2d)[0]
    # sp2 = predictor(img2d, rect2)
    # points2 = [(p.x, p.y) for p in sp2.parts()]

    img1 = cv2.imread(file_name1)
    img2 = cv2.imread(file_name2)

    img1_warped = np.copy(img2)

    points1 = read_points(face_recognition.face_landmarks(img1)[0])
    points2 = read_points(face_recognition.face_landmarks(img2)[0])
    # points1 = readPoints(filename1 + '.txt')
    # points2 = readPoints(filename2 + '.txt')

    # Find convex hull
    hull1 = []
    hull2 = []

    hull_index = cv2.convexHull(np.array(points2), returnPoints=False)

    for i in range(0, len(hull_index)):
        hull1.append(points1[int(hull_index[i])])
        hull2.append(points2[int(hull_index[i])])

    # Find delanauy traingulation for convex hull points

    size_img2 = img2.shape
    rect = (0, 0, size_img2[1], size_img2[0])

    dt = calculate_delaunay_triangles(rect, hull2)

    if len(dt) == 0:
        quit()

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        warp_triangle(img1, img1_warped, t1, t2)

    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img2.shape, dtype=img2.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))

    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1_warped), img2, mask, center, cv2.NORMAL_CLONE)

    name = f'result{datetime.now()}.jpg'
    name = name.replace(':', '')
    path = os.path.join(path, name)
    if not cv2.imwrite(path, output):
        raise Exception("Could not write image")

    return name
    # cv2.imshow("Face Swapped", output)
    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()
