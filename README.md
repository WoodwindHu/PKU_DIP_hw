# PKU_Digital_Image_Processing_hw
数字图像处理（彭宇新）小作业

## Requirement
```
conda create -n DIP python=3.8
conda activate DIP
conda install -c conda-forge flask opencv pillow numpy matplotlib
```

## Usage
演示界面：
```
conda activate DIP
python app.py
```
运行图像处理源代码：
```
conda activate DIP
python main.py -h
usage: main.py [-h] [-i INPUT] [-m MODE] [-o OUTPUT]

Digital Image Processing 
1) Histogram equalization (mode = Grayscale|RGB|HSI) 
2) Laplacian enhancement (mode = LE) 
3) Morphological transformation (mode = Erosion|Dilation|Closing|Opening)

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path of input image
  -m MODE, --mode MODE  DIP mode: Grayscale|RGB|HSI|LE|Erosion|Dilation|Closing|Opening
  -o OUTPUT, --output OUTPUT
                        output image filename, default = None
```
运行`python main.py -i INPUT -m MODE`即可运行图像处理源代码，其中`INPUT`为输入图片路径，`MODE`为需要进行的图像处理的操作：进行直方图均衡化时`MODE`可以选择`Grayscale`、`RGB`和`HSI`；使用锐化滤波器时`MODE`选择`LE`；进行形态学变换时`MODE`选择`Erosion`、`Dilation`、`Closing`和`Opening`。

运行所有演示图片：
```
conda activate DIP
bash ./run.sh
```
