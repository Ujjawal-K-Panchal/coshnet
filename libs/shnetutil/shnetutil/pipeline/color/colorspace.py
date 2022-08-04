"""
MIT License

Copyright (c) 2018 Jorge Pessoa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from skimage.color import (rgb2lab, rgb2yuv, rgb2ycbcr, lab2rgb, yuv2rgb, ycbcr2rgb,
                           rgb2hsv, hsv2rgb, rgb2xyz, xyz2rgb, rgb2hed, hed2rgb)

def err(type_):
	raise NotImplementedError(f'Color space conversion {type_} not implemented yet')

def convert(input_, type_):
	return {
		'rgb2lab': rgb2lab(input_),
		'lab2rgb': lab2rgb(input_),
		'rgb2yuv': rgb2yuv(input_),
		'yuv2rgb': yuv2rgb(input_),
		'rgb2xyz': rgb2xyz(input_),
		'xyz2rgb': xyz2rgb(input_),
		'rgb2hsv': rgb2hsv(input_),
		'hsv2rgb': hsv2rgb(input_),
		'rgb2ycbcr': rgb2ycbcr(input_),
		'ycbcr2rgb': ycbcr2rgb(input_)
	}.get(type_)
