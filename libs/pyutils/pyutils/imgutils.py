import numpy as np
import matplotlib as plt

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_timestamp():
	return datetime.now().strftime('%y%m%d-%H%M%S')


def imshow(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def surf(Z, cmap='rainbow', figsize=None):
    plt.figure(figsize=figsize)
    ax3 = plt.axes(projection='3d')

    w, h = Z.shape[:2]
    xx = np.arange(0,w,1)
    yy = np.arange(0,h,1)
    X, Y = np.meshgrid(xx, yy)
    ax3.plot_surface(X,Y,Z,cmap=cmap)
    #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap=cmap)
    plt.show()	

'''
# --------------------------------------------
# split large images into small images 
# --------------------------------------------
# Adapted from Kai Zhang (github: https://github.com/cszn)
'''
def patches_from_image(img:ndarray, p_size=16, p_overlap=8):
	w, h = img.shape[:2]
	dtype = img.dtype
	patches = []
	w1 = list(np.arange(0, w-p_size, p_size-p_overlap, dtype=dtype))
	h1 = list(np.arange(0, h-p_size, p_size-p_overlap, dtype=dtype))
	w1.append(w-p_size)
	h1.append(h-p_size)
	# print(w1)
	# print(h1)
	for i in w1:
		for j in h1:
			patches.append(img[i:i+p_size, j:j+p_size,:])

	return patches
