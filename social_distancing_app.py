import numpy as np
import cv2
import imutils
import os
import time
from scipy.spatial.distance import pdist, squareform
yolo = "./yolo-coco/"
mouse_pts = []
scale_w = 1.2 / 2
scale_h = 4 / 2
SOLID_BACK_COLOR = (64, 64, 64)
four_points=[]
d_thresh=0
def plot_lines_between_nodes(warped_points, bird_image, d_thresh):
    p = np.array(warped_points)
    dist_condensed = pdist(p)
    dist = squareform(dist_condensed)

    # breaking safe distance: 6 feet mark
    dd = np.where(dist < d_thresh)
    six_feet_violations = len(np.where(dist_condensed < d_thresh)[0])
    total_pairs = len(dist_condensed)
    danger_p = []
    color_6 = (32, 46, 39)
    lineThickness = 4

    for i in range(int(np.ceil(len(dd[0]) / 2))):
        if dd[0][i] != dd[1][i]:
            point1 = dd[0][i]
            point2 = dd[1][i]

            danger_p.append([point1, point2])
            cv2.line(
                bird_image,
                (p[point1][0], p[point1][1]),
                (p[point2][0], p[point2][1]),
                color_6,
                lineThickness
            )
    # Display Birdeye view
    #cv2.imshow("Bird Eye View", bird_image)
    cv2.waitKey(1)

def plot_points_on_bird_eye_view(frame,birdImg, pedestrian_boxes, M, scale_w, scale_h,pairs):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    node_radius = 5
    color_node = (234, 218, 46)
    thickness_node = -1
    solid_back_color = (41, 41, 41)

    blank_image = np.zeros(
        (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
    )
    blank_image[:] = solid_back_color
    warped_pts = []
    # for i in range(len(pedestrian_boxes)):

    mid_point_y = int(
            (2*pedestrian_boxes[1] + pedestrian_boxes[3] ) / 2
        )
    mid_point_x = int(
            (2*pedestrian_boxes[0]  + pedestrian_boxes[2]) / 2
        )
    print(mid_point_x,mid_point_y)
    pts = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
    # print(f'points {pts}')
    # pts_src = np.array([[[0,0],[0,50],[50,50],[50,0]]])
    # pts_dst = np.array([[[0,0],[0,100],[100,100],[100,0]]])
    # h, status = cv2.findHomography(pts_src,pts_dst)
    # print(f'homography mat {h}')
    for h in pairs:
        pp1 = np.array([[[h[0][0], h[0][1]]]], dtype="float32")
        warped_pp1 = cv2.perspectiveTransform(pp1, M)
        wpp1_pt = (warped_pp1[0][0][0],warped_pp1[0][0][1])

        pp2 = np.array([[[h[1][0], h[1][1]]]], dtype="float32")
        warped_pp2 = cv2.perspectiveTransform(pp2, M)
        wpp2_pt = (warped_pp2[0][0][0],warped_pp2[0][0][1])

        cv2.line(birdImg, wpp1_pt, wpp2_pt, (0, 0, 255), 2)



    warped_pt = cv2.perspectiveTransform(pts, M)
    # warped_pt_scaled = [int(warped_pt[0] * scale_w), int(warped_pt[1] * scale_h)]
    print(f' warped_pt_scaled {warped_pt[0][0]}')

    # warped_pts.append(warped_pt_scaled)
    # print(warped_pt)
    #birdImg[mid_point_y,mid_point_x,:] = color_node
    
    
    cv2.circle(
            birdImg,
           (warped_pt[0][0][0],warped_pt[0][0][1]),
            node_radius,
            color_node,
            thickness_node
        )
    
    return warped_pts, birdImg


def get_camera_perspective(img, src_points):
    IMAGE_H = img.shape[0]
    IMAGE_W = img.shape[1]
    src = np.float32(np.array(src_points))
    dst = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return M, M_inv


def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the frame zero of the video that will be warped
    # Used to mark 2 points on the frame zero of the video that are 6 feet away
    cnt=0
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(image, (x, y), 10, (0, 255, 255), 10)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        print("Point detected")
        print(mouse_pts)
        cnt=cnt+1


def Check(a,  b):
    # print(a)
    # print(b)
    dist = ((a[0] - b[0]) ** 2 + 550 / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
    calibration = (a[1] + b[1])    
    if 0 < dist < 0.45 * calibration:
        return True
    else:
        return False

def Setup(yolo):
    global net, ln, LABELS
    weights = os.path.sep.join([yolo, "yolov3.weights"])
    config = os.path.sep.join([yolo, "yolov3.cfg"])
    labelsPath = os.path.sep.join([yolo, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")  
    net = cv2.dnn.readNetFromDarknet(config, weights)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]



def ImageProcess(image,M,Minv):
    global processedImg
    (H, W) = (None, None)
    frame = image.copy()
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    starttime = time.time()
    layerOutputs = net.forward(ln)
    stoptime = time.time()
    #print("Video is Getting Processed at {:.4f} seconds per frame".format((stoptime-starttime))) 
    confidences = []
    outline = []
    cnt=0
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            maxi_class = np.argmax(scores)
            confidence = scores[maxi_class]
            if LABELS[maxi_class] == "person":
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    outline.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))

    box_line = cv2.dnn.NMSBoxes(outline, confidences, 0.5, 0.3)

    if len(box_line) > 0:
        cnt=0
        flat_box = box_line.flatten()
        pairs = []
        center = []
        status = [] 
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])
            status.append(False)

        for i in range(len(center)):
            for j in range(len(center)):
                close = Check(center[i], center[j])

                if close:
                    pairs.append([center[i], center[j]])
                    status[i] = True
                    status[j] = True
        index = 0
        birdimg = np.zeros((frame.shape[0],frame.shape[1],3),np.uint8)
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            
            if status[index] == True:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
                cnt=cnt+1
            elif status[index] == False:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            index += 1
            for h in pairs:
                cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
            
            #cv2.putText(frame, 'Number of people at risk'+ str(len(pairs)), (10,450))
            font = cv2.FONT_HERSHEY_SIMPLEX

            warped_pts, bird_image = plot_points_on_bird_eye_view(
                frame, birdimg,outline[i], M, scale_w, scale_h,pairs
            )
            
        cv2.putText(frame, 'Number of people at risk:'+ str(cnt), (10,50), font, 1, (49, 255, 255), 2, cv2.LINE_AA)

            #cv2.putText(frame,  str(cnt), (10,50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Bird_eye",np.hstack([frame,birdimg]))
            # plot_lines_between_nodes(
            #     warped_pts, bird_image, d_thresh)
  
      
    processedImg = bird_image.copy()
    

create = None
frameno = 0
filename = "./test_video.mp4"
opname = "Desktop/output.mp4"
cap = cv2.VideoCapture(filename)
# cap.set(cv2.CAP_PROP_FPS, 20)
# fourcc = cv2.VideoWriter_fourcc(*'MPEG')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

time1 = time.time()
cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)
while(True):

    ret, frame = cap.read()
    # h = cv2.findHomography()
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    if not ret:
        break
    current_img = frame.copy()
    frame = imutils.resize(frame, width=480)
    #video = frame.shape
    frameno += 1
    if (frameno==1):
        Setup(yolo)
        while True:
            image = frame
            cv2.imshow("image", image)
            cv2.waitKey(1)
            if len(mouse_pts) == 5:
                cv2.destroyWindow("image")
                break
            first_frame_display = False
        four_points = mouse_pts
    
    
    # Get perspective
    M, Minv = get_camera_perspective(frame, four_points[0:4])
    pts = src = np.float32(np.array([four_points[4:]]))
    warped_pt = cv2.perspectiveTransform(pts, M)[0]
    # d_thresh = np.sqrt(
    #         (warped_pt[0][0] - warped_pt[1][0]) ** 2
    #         + (warped_pt[0][1] - warped_pt[1][1]) ** 2
    #     )
    # bird_image = np.zeros(
    #         (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
    #     )

    # bird_image[:] = SOLID_BACK_COLOR
    # pts = np.array(
    #     [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32
    # )
    # cv2.polylines(frame, [pts], True, (0, 255, 255), thickness=4)

    current_img = frame
    #cv2.imshow("frame",current_img)
    ImageProcess(current_img,M,Minv)
    Frame = processedImg
    if create is None:
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            create = cv2.VideoWriter('Desktop/test.mp4', fourcc, 60, (Frame.shape[3], Frame.shape[0]))
            cv2.imshow("Frame",Frame)
            fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            create = cv2.VideoWriter(opname, fourcc, 30, (Frame.shape[1], Frame.shape[0]), True)
            cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
    

time2 = time.time()
#print("Completed. Total Time Taken: {} minutes".format((time2-time1)/60))

cap.release()
cv2.destroyAllWindows()


