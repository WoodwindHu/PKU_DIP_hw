import argparse

def argparser():
    parser = argparse.ArgumentParser(description='''Digital Image Processing 
    1) Histogram equalization (mode = Grayscale|RGB|HSI) 
    2) Laplacian enhancement (mode = LE) 
    3) Morphological transformation (mode = Erosion|Dilation|Closing|Opening)''')
    parser.add_argument('-i', '--input', default='ImagesSet/histeq1.jpg', type=str, help='path of input image')
    parser.add_argument('-m', '--mode', default='Grayscale', type=str, help='DIP mode: Grayscale|RGB|HSI|LE|Erosion|Dilation|Closing|Opening')
    parser.add_argument('-o', '--output', default=None, type=str, help='output image filename, default = None')
    return parser