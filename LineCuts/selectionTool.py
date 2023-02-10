import numpy as np
import holoviews as hv
from scipy import interpolate

def getIndices(x,y, x_axis, y_axis, x_step, y_step):
    """Return the index corresponding to the input x,y values with respect to the input x/y axes. 
    For example if x = 1, and the x-axis is [0, 1, 2, 3, 4] then the output x-index is 1

    Inputs:
        x, y: The x, y values for which the output indices correspond to
        x_axis, y_axis: An array containing the values of the x/y axes
        x_step, y_step: Step sizes of the axes. This is a redundant input as they can be calculated from the axes
    Output:
        (x-index, y-index)
        
    """
    xmin = x_axis.min()
    ymax = y_axis.max()
    ix = (x-xmin)/x_step
    iy = -(y-ymax)/y_step
    return (int(ix), int(iy))
    
def cSelects(images):
    """Takes a holoviews Layout of images and displays them alongside a sliced area. By clicking on each full 2D image, the sliced area can be updated. 
    Single clicking allows multiple selections to be made. Double clicking resets all clicks. 
    The label above the sliced area is the average over all points selected. Since this is always the even elements of the Layout, one way to extract the averages values is
    by imageSelectors[2*i+1].label where i goes from 0 to number of input images.
    
    Inputs:
        images: Holoviews Layout of Holoviews images. This is usually made by 'adding' images -- hv.Image(data1) + hv.Image(data2) + hv.Image(data3) + ...
    Output:
        imageSelectors: A holoviews Layout where between each input image another image is added which contains only the selected areas. 
    
    """
    def tapfuncImage(img):
        x_axis = np.unique(img.dimension_values(0))
        y_axis = np.unique(img.dimension_values(1))
        xstep = x_axis[1]-x_axis[0]
        ystep = y_axis[1]-y_axis[0]
        
        dataInit = np.copy(img.data)
        dataSlice = np.copy(img.data)
        dataSlice[:, :] = np.nan
        def tapImage(x, y, x2, y2):
            if None not in [x,y]:
                x,y = img.closest((x,y))
                ix,  iy = getIndices(x ,y, x_axis, y_axis, xstep, ystep)

                dataSlice[iy-1:iy+2, ix-1:ix+2] = dataInit[iy-1:iy+2, ix-1:ix+2]
                circlePoints = np.nanmean(dataSlice)
            elif None not in [x2, y2]:
                dataSlice[:, :] = np.nan
            return hv.Image(dataSlice).options(title='%s' %(circlePoints,)).redim.range(z=(np.min(img.data), np.max(img.data))).relabel('%s' % (circlePoints,))
        return tapImage
    image = images[0]
    
    posxy = hv.streams.SingleTap(source=image, x=0, y=0, transient = True)
    posxy2 = hv.streams.DoubleTap(source=image, rename={'x': 'x2', 'y': 'y2'}, transient = True)
    tap_dmap = hv.DynamicMap(tapfuncImage(image), streams = [posxy, posxy2])
    imageSelectors = image+tap_dmap
    for i in range(len(images)):
        if i == 0:
            continue
        image = images[i]

        posxy = hv.streams.SingleTap(source=image, x=0, y=0, transient = True)
        posxy2 = hv.streams.DoubleTap(source=image, rename={'x': 'x2', 'y': 'y2'}, transient = True)
        tap_dmap = hv.DynamicMap(tapfuncImage(image), streams = [posxy, posxy2])
        imageSelectors += image+tap_dmap
    return imageSelectors