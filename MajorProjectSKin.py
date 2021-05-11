import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
from pickle import load
from numpy import argmax

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
from matplotlib import pyplot as plt


def extractSkin(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    #lower_threshold = np.array([0, 10, 60], dtype=np.uint8)
    #upper_threshold = np.array([20, 150, 255], dtype=np.uint8)
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)


def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        # make the dictionay of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation

def getSkinTone(colorInformation):
    skinToneColor = sum(colorInformation[0]['color'])
    if skinToneColor < 350:
        userSkinTone = "Dark"
        return userSkinTone
    elif skinToneColor > 350 and skinToneColor < 450 :
        userSkinTone = "Medium"
        return userSkinTone
    else:
        userSkinTone = "Light"
    return userSkinTone

def extractDominantColor(image, number_of_colors=5, hasThresholding=False):

    # Quick Fix Increase cluster counter to neglect the black
    if hasThresholding == True:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img)

    print("Cluster Centers :")
    print(estimator.cluster_centers_)
    print("Cluster Index :")

    print(estimator.labels_)
    #plt.scatter(img[:, 0], img[:, 1], c=estimator.labels_, cmap='rainbow')
    #plt.scatter(estimator.cluster_centers_[:, 0], estimator.cluster_centers_[:, 1], color='black')

    # Get Colour Information
    colorInformation = getColorInformation(estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation



def plotColorBar(colorInformation):
    # Create a 500x100 black image
    color_bar = np.zeros((100, 500, 3), dtype="uint8")

    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int, (x['color'])))

        cv2.rectangle(color_bar, (int(top_x), 0),
                      (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar


def prety_print_data(color_info):
    for x in color_info:
        print(pprint.pformat(x))
        print()


def compute(img, min_percentile, max_percentile):
    """Calculate the quantile, the purpose is to remove the abnormal situation at both ends. """


    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel


def aug(src):
    """Image brightness enhancement"""
    if get_lightness(src) < 130:
        print("The brightness of the picture is not sufficient, so enhancement is required.")

    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)

    # Remove values ​​outside the quantile range
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    # Stretch the quantile range from 0 to 255. 255*0.1 and 255*0.9 are taken here because pixel values ​​may overflow, so it is best not to set it to 0 to 255.
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)
   

   


    return out


def get_lightness(src):
    # Calculate brightness
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    return lightness



def vp_start_gui():
    global root
    root = Tk()


    root.title("Skin Segementation and Skin Tone Detection")

    canvas1 = tk.Canvas(root, width=1000, height=800, relief='raised')
    canvas1.pack()

    label1 = tk.Label(root, text='Skin Segementation and Skin Tone Detection')
    label1.config(font=('helvetica', 14))
    canvas1.create_window(500, 30, window=label1)

    label2 = tk.Label(root, text='Upload Your Picture')
    label2.config(font=('helvetica', 10))
    canvas1.create_window(500, 80, window=label2)

    label7 = tk.Label(root, text='Anup Poudel')
    label7.config(font=('helvetica', 10))
    canvas1.create_window(100, 100, window=label7)

    label8 = tk.Label(root, text='Ashish Sharma')
    label8.config(font=('helvetica', 10))
    canvas1.create_window(100, 120, window=label8)

    label9 = tk.Label(root, text='Krishna Thapa')
    label9.config(font=('helvetica', 10))
    canvas1.create_window(100, 140, window=label9)

    label10 = tk.Label(root, text='Shreya Tiwari')
    label10.config(font=('helvetica', 10))
    canvas1.create_window(100, 160, window=label10)

    label11 = tk.Label(root, text='Developers:')
    label11.config(font=('helvetica', 10, 'bold'))
    canvas1.create_window(100, 80, window=label11)

    def openfn():
        filename = filedialog.askopenfilename(title='open')
        return filename

    def open_img():
        x = openfn()
        img = cv2.imread(x)
        img = aug(img)
        image = img
        # cv2.imwrite('out2.png', img)

        # Resize image to a width of 250
        image = imutils.resize(image, width=250)

        # Show image
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        # plt.show()

        # Apply Skin Mask
        skin = extractSkin(image)

        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
        plt.title("Thresholded  Image")
        # plt.show()

        # Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors
        dominantColors = extractDominantColor(skin, hasThresholding=True)

        # Show in the dominant color information
        print("Color Information")

        prety_print_data(dominantColors)

        skinTone = getSkinTone(dominantColors)
        print(skinTone)
        plt.subplot(2, 2, 3)
        plt.axis("off")
       # plt.text("Skin Tone")
        plt.text(0.5, 0.6, "Your Skin Tone is:", size=12, ha="center")
        plt.text(0.5, 0.5, skinTone, size=12, ha="center")
        #plt.title(skinTone)


        # Show in the dominant color as bar
        print("Color Bar")

        colour_bar = plotColorBar(dominantColors)
        plt.subplot(2, 2, 4)
        plt.axis("off")
        plt.imshow(colour_bar)
        plt.title("Color Bar")

        plt.tight_layout()
        plt.show()
        
        label3 = tk.Label(root, text='The Image :', font=('helvetica', 10))

        canvas1.create_window(500, 190, window=label3)
        img = Image.open(x)
        img = img.resize((350, 350), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(root, image=img)
        panel.image = img
        canvas1.create_window(500, 390, window=panel)

        #label5 = tk.Label(root, text='Caption for the Image:', font=('helvetica', 10))
        #canvas1.create_window(500, 590, window=label5)

        #label6 = tk.Label(root, text=final, font=('helvetica', 10, 'bold'))
        #canvas1.create_window(500, 620, window=label6)
        #final = ""

    button1 = tk.Button(root, text='Open Image', command=open_img, bg='palegreen2', fg='black',
                        font=('helvetica', 9, 'bold'))
    canvas1.create_window(500, 130, window=button1)

    button2 = tk.Button(root, text='Quit', command=root.destroy, bg='lightskyblue2', fg='black', font=('helvetica', 9, 'bold'))
    canvas1.create_window(900, 680, window=button2)

    button3 = tk.Button(root, text='Refresh', command=refresh, bg='lightsteelblue2', fg='black', font=('helvetica', 9, 'bold'))
    canvas1.create_window(100, 680, window=button3)

    root.mainloop()


if __name__ == '__main__':
    def refresh():
        root.destroy()
        vp_start_gui()


    vp_start_gui()