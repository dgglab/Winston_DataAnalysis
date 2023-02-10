import numpy as np
import holoviews as hv
from scipy.optimize import curve_fit
from scipy import signal
from scipy import interpolate
import scipy.ndimage
from scipy import stats
import pandas as pd

def lineCut(data, x0, y0, x1, y1, dVx = False, dVy = False):
    """Input 2d array data and take line cut specified by two points. 
    Intermediate points that don't fall exactly on grid are calculated through cubic spline interpolation
    
    Inputs:
        data: 2D array to have line cut taken of
        x0: x pixel coordinate of where line cut starts from
        y0: y pixel coordinate of where line cut starts from
        x1: x pixel coordinate of where line cut ends
        y1: y pixel coordinate of where line cut ends
        
    Output:
        imageCut: Combined Holoviews object of plotted data with line overlayed displaying where line cut taken
        lineCut: Z values along line cut
        
    """
    xcoords = range(data.shape[1])
    ycoords = range(data.shape[0])
    

    image = hv.Image((xcoords, ycoords,data)).options(height=300, width=400,colorbar=True, tools=['hover'])
    length = np.hypot(x1-x0, y1-y0)
    numPoints = int(np.round(length,0))
    x, y = np.linspace(x0, x1, numPoints), np.linspace(y0, y1, numPoints)
    xcoords = np.array([xcoords[x0], xcoords[x1]])
    ycoords = np.array([ycoords[y0], ycoords[y1]])
    # Extract the values along the line
    zi = scipy.ndimage.map_coordinates(data, np.vstack((y,x)))#dataG[y.astype(np.int), x.astype(np.int)]
    if dVx and dVy:
        pixelSpacing = pixelToVoltage(1, dVx, dVy, x0, y0, x1, y1)
        numPixels = len(zi)
        lineCutAxis = np.linspace(-pixelSpacing*(numPixels-1)/2, pixelSpacing*(numPixels-1)/2, numPixels)
        return (image*hv.Curve((xcoords, ycoords)) 
            + hv.Curve((lineCutAxis,zi), kdims = 'Plunger', vdims = 'G')).options(shared_axes=False), zi, lineCutAxis
    return (image*hv.Curve((xcoords, ycoords)) 
            + hv.Curve(zi, kdims = 'Vdiag (pixel units)', vdims = 'G')).options(shared_axes=False), zi

def multiplelineCut(data, length, centers, slope=1, dVx = False, dVy = False):
    """Input 2d array data and take line cut specified by two points. 
    Intermediate points that don't fall exactly on grid are calculated through cubic spline interpolation
    
    Inputs:
        data: 2D array to have line cut taken of
        x0: x pixel coordinate of where line cut starts from
        y0: y pixel coordinate of where line cut starts from
        x1: x pixel coordinate of where line cut ends
        y1: y pixel coordinate of where line cut ends
        
    Output:
        imageCut: Combined Holoviews object of plotted data with line overlayed displaying where line cut taken
        lineCut: Z values along line cut
        
    """
    xcoords = range(data.shape[1])
    ycoords = range(data.shape[0])

    image = hv.Image((xcoords, ycoords,data)).options(height=300, width=400,colorbar=True, tools=['hover'])
    angle = np.arctan(slope)
    numPoints = int(np.round(length,0))
    zi = np.zeros(numPoints)
    lines = hv.Curve((centers[0][0], centers[0][1]))

    for i in range(centers.shape[0]):
        xcoords = range(data.shape[1])
        ycoords = range(data.shape[0])
        center = centers[i]
        #print(center)
        x1, y1 = tuple(map(sum, zip(center, (np.cos(angle)*length/2, np.sin(angle)*length/2))))
        x0, y0 =  tuple(map(sum, zip(center, (-np.cos(angle)*length/2, -np.sin(angle)*length/2))))

        x0, x1, y0, y1 = np.round(np.array([x0, x1, y0, y1]), 0)
        x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
        x, y = np.linspace(x0, x1, numPoints), np.linspace(y0, y1, numPoints)
        xcoords = np.array([xcoords[x0], xcoords[x1]])
        ycoords = np.array([ycoords[y0], ycoords[y1]])
        lines *= hv.Curve((xcoords, ycoords))#.options(color='blue')
        # Extract the values along the line
        zi += scipy.ndimage.map_coordinates(data, np.vstack((y,x)))#dataG[y.astype(np.int), x.astype(np.int)]
    zi = zi/centers.shape[0]
    if dVx and dVy:
        pixelSpacing = pixelToVoltage(1, dVx, dVy, x0, y0, x1, y1)
        numPixels = len(zi)
        lineCutAxis = np.linspace(-pixelSpacing*(numPixels-1)/2, pixelSpacing*(numPixels-1)/2, numPixels)
        return (image*lines
            + hv.Curve((lineCutAxis,zi), kdims = 'Plunger', vdims = 'G')).options(shared_axes=False), zi, lineCutAxis
    return (image*lines
            + hv.Curve(zi, kdims = 'Vdiag (pixel units)', vdims = 'G')).options(shared_axes=False), zi

def pixelToVoltage(pixelLength, dVx, dVy, x0, y0, x1, y1):
    """Converts a length in pixel coordinates to gate voltage units
    
    Inputs:
        pixelWidth: length in pixel coordinates to be converted
        dVx: Gate voltage spacing for each pixel on the x-axis
        dVy: Gate voltage spacing for each pixel on the y-axis
        x0, y0, x1, y1: Same as in lineCut -- starting and ending 
            coordinates for line cut which this length is being taken from.
            This is necessary because if dVx != dVy, the length of the line cut in voltage coordinates is not
            simply a multiplicative factor of the lenght of the line in pixel coordinates
            
    """
    
    lineLengthVoltage = np.sqrt(((x1-x0)*dVx)**2 + ((y1-y0)*dVy)**2)
    length = np.hypot(x1-x0, y1-y0)
    numPoints = int(np.round(length,0))
    Vspacing = lineLengthVoltage/(numPoints-1)
    return pixelLength*Vspacing
    
def lineCutAveraging(xdata, data, returnMAD = False):
    """Returns the median values of data at the same xdata points. Useful to average multiple line cuts.
    If the mean is desired, change nanmedian to nanmean.
    
    Inputs:
        xdata: array of x data points
        data: y values corresponding to each x point
        returnMAD: Boolean specifying whether the median absolute deviation should be returned
        
    Output:
        A tuple of (xdata, median(data)
    """
    if not isinstance(data, np.ndarray): 
        data = np.array(data)
    if not isinstance(xdata, np.ndarray): 
        xdata = np.array(xdata)
    
    xdata = np.round(xdata, 3) #Avoid floating point errors
    xvalues = np.unique(xdata)
    xdataAvg = []
    dataAvg = []
    MAD = []
    for xvalue in xvalues:
        xdataAvg.append(xvalue)
        dataAvg.append(np.nanmedian(data[xdata==xvalue]))
        MAD.append(stats.median_abs_deviation(data[xdata==xvalue], nan_policy = 'omit'))

    if returnMAD:
        return xdataAvg, dataAvg, MAD
    return xdataAvg, dataAvg