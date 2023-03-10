import kociemba
import pandas as pd
import cv2
import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import serial
import time

state = ""
face = ""
mean_value_package = []
current_package = []


def record_data(file_name, package):
    df = pd.read_csv(file_name)
    if df.empty:
        for mean in package:
            df = df.append({'B': mean[0], 'G': mean[1], 'R': mean[2], 'Color': mean[3]}, ignore_index=True)
    else:
        for mean in package:
            df = df.append({'B': mean[0], 'G': mean[1], 'R': mean[2], 'Color': mean[3]}, ignore_index=True)

    df.to_csv(file_name, index=False)


def get_mean_value(contours, src_img):
    # creating the mask from the contour
    contour_mask = np.zeros(src_img.shape, src_img.dtype)
    cv2.drawContours(contour_mask, contours, -1, (255, 255, 255), -1)

    contour_mask = cv2.cvtColor(contour_mask, cv2.COLOR_BGR2GRAY)

    # taking all the pixel from the src image within the mask
    result = cv2.bitwise_and(src_img, src_img, mask=contour_mask)
    contour_mean = cv2.mean(src_img, contour_mask)

    return contour_mean


def color_to_text(value):
    if value == 1:
        return "R"
    if value == 2:
        return "O"
    if value == 3:
        return "B"
    if value == 4:
        return "G"
    if value == 5:
        return "W"
    if value == 6:
        return "Y"


def sort_face(points_list):
    points_list.sort(key=lambda point: point[1])
    result = ""

    for i in range(0, len(points_list), 3):

        array_sorting_x = []

        for j in range(i, i + 3):
            array_sorting_x.append(points_list[j])

        array_sorting_x.sort(key=lambda point: point[0])

        for k in array_sorting_x:
            result += k[2]

        # result += '\n'
    return result


def snap_vertical(ser):
    ser.write(bytes("0", 'utf-8'))
    time.sleep(0.5)


def snap_horizontal(ser):
    ser.write(bytes("1", 'utf-8'))
    time.sleep(0.5)


def solve(ser):
    global state
    global mean_value_package

    record_data('data.csv', mean_value_package)
    print(mean_value_package)
    mean_value_package = []


    if len(state) == 54:
        print(state)
        state = state.replace('W', 'U')
        state = state.replace('R', 'F')
        state = state.replace('B', 'R')
        state = state.replace('G', 'L')
        state = state.replace('Y', 'D')
        state = state.replace('O', 'B')

        solve_code = kociemba.solve(state)
        ser.write(bytes(solve_code, 'utf-8'))

    #ser.write(bytes("2", 'utf-8'))
    time.sleep(0.5)
    state = ""
    face = ""


def save_face():
    global state
    global face
    global mean_value_package
    global current_package
    state += face
    mean_value_package = mean_value_package + current_package


# Define function to show frame
def show_frames(model, cap, label, win, text):
    # Get the latest frame and convert into Image
    ret, frame = cap.read()
    h, w, c = frame.shape
    frame, text1 = image_processing_and_predict(frame, model, win, text)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.place(x=(800 - w) / 2, y=20)
    # Repeat after an interval to capture continiously
    label.after(20, show_frames, model, cap, label, win, text1)


def image_processing_and_predict(frame, model, win, text):
    LOWER_AREA_THRESHOLD = 3000
    UPPER_AREA_THRESHOLD = 10000
    global state
    global face
    global mean_value_package
    global current_package

    original_frame = frame.copy()

    # convert to gray image for image processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("original", gray)
    # axarr[0,0].imshow(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

    # bluring an image
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # run canny function for edge detection
    # 20 40
    gray = cv2.Canny(gray, 20, 40)

    # image dilation
    gray = cv2.dilate(gray, (5, 5), iterations=1)

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    i = 0
    contour_id = 0
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
                blob_color = np.array(cv2.mean(frame[y:y + h, x:x + w])).astype(int)
                # print(get_mean_value([contour], frame1))
                correct_contour.append(contour)

                cv2.drawContours(frame, [contour], 0, (255, 255, 0), 2)
                cv2.drawContours(frame, [approx], 0, (255, 255, 0), 2)

    points_list = []
    package = []
    for contour in correct_contour:
        mean = get_mean_value([contour], original_frame)

        predict = model.predict([[mean[0], mean[1], mean[2]]])
        # print(predict)
        package.append([mean[0], mean[1], mean[2], predict[0]])

        color = color_to_text(predict[0])

        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            points_list.append((cx, cy, color))

            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            cv2.putText(frame, color, (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    if len(points_list) == 9:
        face = sort_face(points_list)
        current_package = package

    #cv2.putText(frame, face, (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

    if len(frame) > 0:
        # cv2.putText(frame, face, (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)
        text.destroy()
        text = tk.Label(win, text=face, font=('Arial', 18))
        text.place(x=325, y=550)

    return frame, text


def main():
    # read and train
    color_data = pd.read_csv('data.csv')
    color_data = color_data.drop(['Unnamed: 0'], axis=1)

    features = ['B', 'G', 'R']
    X_feature = color_data[features]
    target = color_data['Color']

    color_model = KNeighborsClassifier(n_neighbors=7)
    color_model.fit(X_feature.values, target)

    ser = serial.Serial(port='COM9', baudrate=115200, timeout=1000)

    # Create an instance of TKinter Window or frame
    win = tk.Tk()

    # Set the size of the window
    win.geometry("800x800")

    # Create a Label to capture the Video frames
    label = tk.Label(win)
    label.grid(row=0, column=0)

    # canvas = tk.Canvas(win, width=300, height=50, bg="gray")
    text = tk.Label(win, text=face, font=('Arial', 18))
    text.place(x=325, y=550)

    # canvas.create_text(30, 50, text="HELLO WORLD", fill="black", font=('Helvetica 15 bold'))
    # canvas.pack(padx=100, pady=400)
    # canvas.place(relx= 100, rely = 600)



    #snap_ver =
    tk.Button(win, text="Snap Vertical", bg="gray", fg="white", command=lambda: snap_vertical(ser)).place(x=200, y=650, width=100)
    #snap_hori =
    tk.Button(win, text="Snap Horizontal", bg="gray", fg="white", command=lambda: snap_horizontal(ser)).place(x=200, y=700, width=100)
    #record =
    tk.Button(win, text="Record", bg="gray", fg="white", command=save_face).place(x=500, y=650, width=100)
    #solve_button =
    tk.Button(win, text="Solve", bg="gray", fg="white", command=lambda: solve(ser)).place(x=500, y=700, width=100)

    cap = cv2.VideoCapture(0)

    show_frames(color_model, cap, label, win, text)

    win.mainloop()
    #ser.close()


if __name__ == "__main__":
    main()
