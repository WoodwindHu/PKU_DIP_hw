# Histogram equalization
python main.py -i ImagesSet/histeq1.jpg -m Grayscale
python main.py -i ImagesSet/histeq2.jpg -m Grayscale
python main.py -i ImagesSet/histeq3.jpg -m Grayscale
python main.py -i ImagesSet/histeq4.jpg -m Grayscale
python main.py -i ImagesSet/histeqColor.jpg -m RGB
python main.py -i ImagesSet/histeqColor.jpg -m HSI
# Laplacian enhancement
python main.py -i ImagesSet/moon.tif -m LE 
# Morphological transformation
python main.py -i ImagesSet/word_bw.bmp -m Erosion
python main.py -i ImagesSet/word_bw.bmp -m Dilation
python main.py -i ImagesSet/word_bw.bmp -m Closing
python main.py -i ImagesSet/word_bw.bmp -m Opening
