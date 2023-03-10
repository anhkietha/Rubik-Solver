import numpy as np
import cv2
import pandas as pd

LOWER_AREA_THRESHOLD = 3000
UPPER_AREA_THRESHOLD = 10000

RED = 1
ORANGE = 2
BLUE = 3
GREEN = 4
WHITE = 5
YELLOW = 6

#take the capture image from the camera
cap = cv2.VideoCapture(0)

#making the slide for finding the threshold
def nothing(x):
    pass

def get_mean_value(contours,src_img):
    #creating the mask from the contour
    contour_mask = np.zeros(src_img.shape, src_img.dtype)
    cv2.drawContours(contour_mask, contours, -1, (255, 255, 255), -1)

    contour_mask = cv2.cvtColor(contour_mask, cv2.COLOR_BGR2GRAY)

    #taking all the pixel from the src image within the mask
    result = cv2.bitwise_and(src_img, src_img, mask=contour_mask)
    contour_mean = cv2.mean(src_img, contour_mask)

    return contour_mean

#this is a function for recording data only
#the name of the target feature will be changed for each color
#for example, I will record the red color only for the first time so that
#the target value will be changed to 1
def record_data(file_name, mean_value):
    df = pd.read_csv(file_name)
    print(df)
    if df.empty:
        #df_rows = []
        print("here")
        for mean in mean_value:
            df = df.append({'B': mean[0], 'G': mean[1], 'R': mean[2], 'Color': WHITE}, ignore_index=True)
    else:
        for mean in mean_value:
            df = df.append({'B': mean[0], 'G': mean[1], 'R': mean[2], 'Color': WHITE}, ignore_index=True)

    df.to_csv(file_name, index=False)



def main():
    #create a window for the trackbar
    cv2.namedWindow('image')

    #create a trackbar for chosing canny parameters
    cv2.createTrackbar('canny-1', 'image', 0, 500, nothing)
    cv2.createTrackbar('canny-2', 'image', 0, 500, nothing)

    #the is a varible for record all the mean value
    mean_value_array = []

    while True:
        #read the camera
        ret, frame1 = cap.read()

        original_frame = frame1.copy()

        #convert to gray image for image processing
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        cv2.imshow("original", gray)

        #bluring an image
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        #getting the value from the trackbar for the canny function
        canny_1 = cv2.getTrackbarPos('canny-1', 'image')
        canny_2 = cv2.getTrackbarPos('canny-2', 'image')

        #run canny function for edge detection
        # 20 40
        gray = cv2.Canny(gray, canny_1, canny_2)

        #image dilation
        gray = cv2.dilate(gray, (5, 5), iterations=1)

        cv2.imshow("canny", gray)


        # contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        i = 0
        contour_id = 0
        # print(len(contours))
        count = 0


        correct_contour = []

        for contour in contours:
            A1 = cv2.contourArea(contour)
            contour_id = contour_id + 1

            if A1 < UPPER_AREA_THRESHOLD and A1 > LOWER_AREA_THRESHOLD:
                perimeter = cv2.arcLength(contour, True)
                epsilon = 0.01 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                hull = cv2.convexHull(contour)
                if cv2.norm(((perimeter / 4) * (perimeter / 4)) - A1) < 2000:
                    # if cv2.ma
                    count = count + 1
                    x, y, w, h = cv2.boundingRect(contour)
                    # cv2.rectangle(bgr_image_input, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    # cv2.imshow('cutted contour', bgr_image_input[y:y + h, x:x + w])
                    val = (50 * y) + (10 * x)
                    blob_color = np.array(cv2.mean(frame1[y:y + h, x:x + w])).astype(int)
                    #print(get_mean_value([contour], frame1))
                    correct_contour.append(contour)

                    cv2.drawContours(frame1, [contour], 0, (255, 255, 0), 2)
                    cv2.drawContours(frame1, [approx], 0, (255, 255, 0), 2)

        cv2.imshow("camera", frame1)

        if len(correct_contour) >= 7 :
            for contour in correct_contour:
                mean_value_array.append(get_mean_value([contour], original_frame))

                if len(mean_value_array) == 50:
                    #pass
                    record_data('data.csv', mean_value_array)



        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()