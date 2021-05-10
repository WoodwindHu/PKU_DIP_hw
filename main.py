import cv2 
import numpy as np
import os
from algorithm import Morphological_Transformation, Histogram_Equalization, Laplacian_Filter, argparser

def main(args):
    img = cv2.imread(args.input, 1)
    if (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 1] == img[:, :, 2]).all() and (img[:, :, 0] == img[:, :, 2]).all():
        img = cv2.imread(args.input, 0)

    if args.mode == 'Erosion' or args.mode == 'Dilation' or args.mode == 'Closing' or args.mode == 'Opening':       
        kernel = np.random.randint(0, 2, (3, 3)).astype(np.uint8)
        print(kernel)

    if args.mode == 'Grayscale':
        result = Histogram_Equalization.MyHE(img)[0]
    elif args.mode == 'RGB':
        result = Histogram_Equalization.MyHE_RGB(img)
    elif args.mode == 'HSI':
        HSI_img = Histogram_Equalization.rgb2hsi(img)
        result = Histogram_Equalization.MyHE_HSI(HSI_img)
    elif args.mode == 'LE':
        result = Laplacian_Filter.MyLaplacianEnhance(img)
    elif args.mode == 'Erosion':
        result = Morphological_Transformation.MyErosion(img, kernel)
    elif args.mode == 'Dilation':
        result = Morphological_Transformation.MyDilation(img, kernel)
    elif args.mode == 'Closing':
        result = Morphological_Transformation.MyClosing(img, kernel)
    elif args.mode == 'Opening':
        result = Morphological_Transformation.MyOpening(img, kernel)
    else:
        print("Mode error!")
        return 
    
    if args.output == None:
        dir, filename = os.path.split(args.input)
        dir = os.path.join(dir, args.mode)
        if not os.path.exists(dir):
            os.makedirs(dir)
        args.output = os.path.join(dir, filename)
    
    cv2.imwrite(args.output, result)
    print('save to', args.output)


if __name__ == '__main__':
    parser = argparser.argparser()
    args = parser.parse_args()
    main(args)