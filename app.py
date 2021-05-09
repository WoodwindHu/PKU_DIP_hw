# web-app for API image manipulation

from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image
#从文件夹algorithm引入算法接口（Morphological_Transformation为形态学处理的代码）
from algorithm import Morphological_Transformation, Fast_Fourier_Transform, Histogram_Equalization, Laplacian_Filter
import numpy as np
import cv2
import time
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# default access page
@app.route("/")
def main():
    return render_template('new_index.html')


# upload selected image and forward to processing page
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/temp_images/')

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg") or (ext == ".png") or (ext == ".bmp") or (ext == ".tif") or (ext == ".npy"):
        print("File accepted")
    else:
        return render_template("new_error.html", message="The selected file is not supported"), 400


    # save file
    local_time = time.localtime(time.time())

    filename= "filename-{}-{}".format(local_time,filename)
    # filename= "filename-{}".format(filename)

    destination = "/".join([target, filename])
    if os.path.isfile(destination):
        os.remove(destination)
    print("File saved to:", destination)
    upload.save(destination)

    if ext == ".tif":
        img = cv2.imread(destination)
        filename = os.path.splitext(filename)[0]+".png"
        destination = "/".join([target, filename])
        if os.path.isfile(destination):
            os.remove(destination)
        print("File saved to:", destination)
        cv2.imwrite(destination, img)

    # forward to processing page
    return render_template("new_processing.html", image_name=filename)


@app.route("/M_T", methods=["POST"])
def M_T():

    # retrieve parameters from html form
    mode = request.form['mode']
    filename = request.form['image']

    # open and process image
    target = os.path.join(APP_ROOT, 'static/temp_images')
    destination = "/".join([target, filename])

    img = cv2.imread(destination, 1)
    if (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 1] == img[:, :, 2]).all() and (img[:, :, 0] == img[:, :, 2]).all():
        img = cv2.imread(destination, 0)

    print(img.shape, img.dtype)

    kernel = np.random.randint(0, 2, (3, 3)).astype(np.uint8)
    print(kernel)

    
    erosion_img = Morphological_Transformation.MyErosion(img, kernel)

    dilation_img = Morphological_Transformation.MyDilation(img, kernel)

    opening_img = Morphological_Transformation.MyOpening(img, kernel)

    closing_img = Morphological_Transformation.MyClosing(img, kernel)

    erosion_name = "erosion-{}".format(filename)
    destination = "/".join([target, erosion_name])
    if os.path.isfile(destination):
        os.remove(destination)
    cv2.imwrite(destination, erosion_img)

    dilation_name = "dilation-{}".format(filename)
    destination = "/".join([target, dilation_name])
    if os.path.isfile(destination):
        os.remove(destination)
    cv2.imwrite(destination, dilation_img)

    opening_name = "opening-{}".format(filename)
    destination = "/".join([target, opening_name])
    if os.path.isfile(destination):
        os.remove(destination)
    cv2.imwrite(destination, opening_img)

    closing_name = "closing-{}".format(filename)
    destination = "/".join([target, closing_name])
    if os.path.isfile(destination):
        os.remove(destination)
    cv2.imwrite(destination, closing_img)

    kernel_name = "kernel-{}".format(filename)
    destination = "/".join([target, kernel_name])
    if os.path.isfile(destination):
        os.remove(destination)
    kernel = cv2.resize(kernel*255, (256, 256),
                        interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(destination, kernel)

    return render_template("MT_result.html", original_name=filename, kernel_name=kernel_name, erosion_name=erosion_name, dilation_name=dilation_name, opening_name=opening_name, closing_name=closing_name)

@app.route("/H_E", methods=["POST"])
def H_E():

    # retrieve parameters from html form
    mode = request.form['mode']
    filename = request.form['image']

    # open and process image
    target = os.path.join(APP_ROOT, 'static/temp_images')
    destination = "/".join([target, filename])

    img = cv2.imread(destination, 1)
    if (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 1] == img[:, :, 2]).all() and (img[:, :, 0] == img[:, :, 2]).all():
        img = cv2.imread(destination, 0)

    print(img.shape, img.dtype)

    # check mode
    if mode == 'Grayscale':
        if len(img.shape) == 3:
            return render_template("new_error.html", message="Invalid mode (color)"), 400
        img, T, h, h_result = Histogram_Equalization.MyHE(img)
        result_name = "{}-{}".format(mode, filename)
        destination = "/".join([target, result_name])
        if os.path.isfile(destination):
            os.remove(destination)
        cv2.imwrite(destination, img)

        hist_name = "hist-{}-{}".format(mode, filename)
        destination = "/".join([target, hist_name])
        if os.path.isfile(destination):
            os.remove(destination)
        plt.bar(range(256), h, label='h')
        plt.bar(range(256), h_result, label='h_result')
        plt.savefig(destination)
        plt.cla()

        HE_name = "HE-{}-{}".format(mode, filename)
        destination = "/".join([target, HE_name])
        if os.path.isfile(destination):
            os.remove(destination)
        plt.plot(range(256), T)
        plt.savefig(destination)

        return render_template("HE_Grayscale_result.html", original_name=filename, result_name=result_name, hist_name=hist_name, HE_name=HE_name)
    elif mode == 'Color':
        if len(img.shape) == 2:
            return render_template("new_error.html", message="Invalid mode (grayscale)"), 400
        img_rgb = Histogram_Equalization.MyHE_RGB(img)
        rgb_name = "RGB-{}-{}".format(mode, filename)
        destination = "/".join([target, rgb_name])
        if os.path.isfile(destination):
            os.remove(destination)
        cv2.imwrite(destination, img_rgb)

        HSI = Histogram_Equalization.rgb2hsi(img)
        I = (HSI[:,:,2]).astype(np.uint8)
        intensity_name = "intensity-{}-{}".format(mode, filename)
        destination = "/".join([target, intensity_name])
        if os.path.isfile(destination):
            os.remove(destination)
        cv2.imwrite(destination, I)

        img_hsi = Histogram_Equalization.MyHE_HSI(HSI)
        hsi_name = "HSI-{}-{}".format(mode, filename)
        destination = "/".join([target, hsi_name])
        if os.path.isfile(destination):
            os.remove(destination)
        cv2.imwrite(destination, img_hsi)


        return render_template("HE_Color_result.html", original_name=filename, rgb_name=rgb_name, hsi_name=hsi_name, intensity_name=intensity_name)
    else:
        return render_template("new_error.html", message="Invalid mode (vertical or horizontal)"), 400


@app.route("/L_F", methods=["POST"])
def L_F():

    # retrieve parameters from html form
    mode = request.form['mode']
    filename = request.form['image']

    # open and process image
    target = os.path.join(APP_ROOT, 'static/temp_images')
    destination = "/".join([target, filename])

    img = cv2.imread(destination, 1)
    if (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 1] == img[:, :, 2]).all() and (img[:, :, 0] == img[:, :, 2]).all():
        img = cv2.imread(destination, 0)

    print(img.shape, img.dtype)

    # check mode
    if len(img.shape) == 3:
        return render_template("new_error.html", message="Invalid image (color)"), 400
    filtered_img_ = Laplacian_Filter.MyLaplacian(img)
    filtered_img = (np.clip(filtered_img_, 0, 255)).astype(np.uint8)
    filtered_name = "Filtered-{}-{}".format(mode, filename)
    destination = "/".join([target, filtered_name])
    if os.path.isfile(destination):
        os.remove(destination)
    cv2.imwrite(destination, filtered_img)

    result = (np.clip(img - filtered_img_, 0, 255)).astype(np.uint8)
    enhanced_name = "Enhanced-{}-{}".format(mode, filename)
    destination = "/".join([target, enhanced_name])
    if os.path.isfile(destination):
        os.remove(destination)
    cv2.imwrite(destination, result)

    return render_template("LF_result.html", original_name=filename, filtered_name=filtered_name, enhanced_name=enhanced_name)
    
    

# retrieve file from 'static/temp_images' directory
@app.route('/static/temp_images/<filename>')
def send_image(filename):
    return send_from_directory("static/temp_images", filename)


if __name__ == "__main__":
    app.run(debug = True)