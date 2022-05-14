
package utils;

import java.util.Arrays;
import java.util.stream.IntStream;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ImageProcessingUtils {
	
	/**
	 * Upsample an image using the nearest neighbors method. Only for the X and Y plane
	 * @param im
	 * 	the image of interest
	 * @param scaleX
	 * 	scale along the X axis
	 * @param scaleY
	 * 	scale factor along the Y axis
	 * @param axesOrder
	 * 	the order in which the axes of the array are
	 */
	public static void upscaleXY(INDArray im, int scaleX, int scaleY, String axesOrder) {
    	if (im.dataType() != DataType.FLOAT)
    		throw new IllegalArgumentException("The input INDArray should be a FLOAT array");
		String nAxesOrder = convertToCompleteAxesOrder(axesOrder);
		long[] shape = arrayToWantedAxesOrderAddOnes(im.shape(), axesOrder, nAxesOrder);
		int xInd = nAxesOrder.toLowerCase().indexOf("x");
		int yInd = nAxesOrder.toLowerCase().indexOf("y");
		long[] targetShape = new long[] {shape[0], shape[1], shape[2], shape[3], shape[4]};
		targetShape[xInd] = shape[xInd] * scaleX;
		targetShape[yInd] = shape[yInd] * scaleY;
		int[] nonXYInds = IntStream.range(0, nAxesOrder.length()).filter(i -> i != xInd && i != yInd).toArray();

		int[] position = new int[5];
		int[] positionNew = new int[5];
		float[] sourceArr = im.data().asFloat();
		float[] targetArr = new float[sourceArr.length * scaleX *scaleY];
		for (int x = 0; x < (int) shape[xInd]; x ++) {
			for (int sx = 0; sx < scaleX; sx ++) {
				int nx = x * scaleX + sx;
				for (int y = 0; y < (int) shape[yInd]; y ++) {
					for (int sy = 0; sy < scaleY; sy ++) {
						int ny = y * scaleY + sy;
						for (int d0 = 0; d0 < nonXYInds[0]; d0 ++) {
							for (int d1 = 0; d1 < nonXYInds[0]; d1 ++) {
								for (int d2 = 0; d2 < nonXYInds[0]; d2 ++) {
									position[xInd] = x;
									position[yInd] = y;
									position[nonXYInds[0]] = d0;
									position[nonXYInds[1]] = d1;
									position[nonXYInds[2]] = d2;
									positionNew[xInd] = nx;
									positionNew[yInd] = ny;
									positionNew[nonXYInds[0]] = d0;
									positionNew[nonXYInds[1]] = d1;
									positionNew[nonXYInds[2]] = d2;
									int pos = getFlatPosOfNDArray(position, shape);
									int nPos = getFlatPosOfNDArray(positionNew, targetShape);
									targetArr[nPos] = sourceArr[pos];
								}
							}
						}
					}
				}
			}
		}
		im.data().setData(targetArr);
    	im.reshape(targetShape);
	}
    
    /**
     * @param image: image to be padded
     * @param padding: number of values padded to the edges of each axis 
     * ((before_1,after_1), … (before_N, after_N))
     * @param value: value to which the padding will be set
     * @param axesOrder
     * 	the order of the dimensions of the array
     * @return padded image of the needed size
     */
    public static void pad(INDArray image, double[][] padding, int value, String axesOrder) {
    	if (image.dataType() != DataType.FLOAT)
    		throw new IllegalArgumentException("The input INDArray should be a FLOAT array");
		String nAxesOrder = convertToCompleteAxesOrder(axesOrder);
		long[] shape = arrayToWantedAxesOrderAddOnes(image.shape(), axesOrder, nAxesOrder);
		int xInd = axesOrder.toLowerCase().indexOf("x");
		int yInd = axesOrder.toLowerCase().indexOf("y");
    	int h = (int) shape[yInd];
    	int w = (int) shape[xInd];
    	int c = (int) shape[axesOrder.toLowerCase().indexOf("c")];
    	int z = (int) shape[axesOrder.toLowerCase().indexOf("z")];
    	int t = (int) shape[axesOrder.toLowerCase().indexOf("b")];
    	int topPad = (int) padding[0][0];
    	int leftPad = (int) padding[1][0];
    	int newH = h + (int) padding[0][0] + (int) padding[0][1];
    	int newW = w + (int) padding[1][0] + (int) padding[1][1];
    	long[] nShape = Arrays.copyOf(shape, shape.length);
    	nShape[xInd] = newW;
    	nShape[yInd] = newH;
    	float[] sourceArr = image.data().asFloat();
    	float[] targetArr = new float[c * t * z * newH * newW];
    	int[] position = new int[5];
    	int[] nPosition = new int[5];
    	for (int cc = 0; cc < c; cc ++) {
    		for (int tt = 0; tt < t; tt ++) {
    			for (int zz = 0; zz < z ; zz ++) {
    				for (int xx = 0; xx < w; xx ++) {
    					for (int yy = 0; yy < h; yy ++) {
    						position[yInd] = yy;
    						position[xInd] = xx;
    						position[axesOrder.toLowerCase().indexOf("c")] = cc;
    						position[axesOrder.toLowerCase().indexOf("z")] = zz;
    						position[axesOrder.toLowerCase().indexOf("b")] = tt;
    						nPosition[yInd] = yy + topPad;
    						nPosition[xInd] = xx + leftPad;
    						nPosition[axesOrder.toLowerCase().indexOf("c")] = cc;
    						nPosition[axesOrder.toLowerCase().indexOf("z")] = zz;
    						nPosition[axesOrder.toLowerCase().indexOf("b")] = tt;
							int pos = getFlatPosOfNDArray(position, shape);
							int nPos = getFlatPosOfNDArray(nPosition, nShape);
							targetArr[nPos] = sourceArr[pos];
    					}
    				}
    			}
    		}
    	}
    	image.data().setData(targetArr);
    	image.reshape(nShape);
    }
    
    /**
     * Method to obtain the index of a position in a flat array obtained from an NDArray
     * @param pos
     * 	array containing positions per axes
     * @param size
     * 	the size of the NDArray per dimension
     * @return the position on a flat array obtained from an NDArray
     */
    public static int getFlatPosOfNDArray(int[] pos, long[] size){
    	int flatPos = 
    			(int) (pos[0] * (size[1] * size[2] * size[3] * size[4])
		+ pos[1] * (size[2] * size[3] * size[4])
		+ pos[2] * (size[3] * size[4])
		+ pos[3] * (size[4])
		+ pos[4]);
    	return flatPos;
    }
    
    /**
     * Convert the array following given axes order into
     *  another long[] which follows the target axes order
     *  The newly added components will be ones.
     * @param size
     * 	original array following the original axes order
     * @param orginalAxes
     * 	axes order of the original array
     * @param targetAxes
     * 	axes order of the target array
     * @return a size array in the order of the tensor of interest
     */
    public static long[] arrayToWantedAxesOrderAddOnes(long[] size, String orginalAxes, String targetAxes) {
    	orginalAxes = orginalAxes.toLowerCase();
    	String[] axesArr = targetAxes.toLowerCase().split("");
    	long[] finalSize = new long[targetAxes.length()];
    	for (int i = 0; i < finalSize.length; i ++) {
    		int ind = orginalAxes.indexOf(axesArr[i]);
    		if (ind == -1) {
    			finalSize[i] = 1;
    		} else {
    			finalSize[i] = size[ind];
    		}
    	}
    	return finalSize;
    }
	
	/**
	 * Create an axesOrder String with 5 axes "xyczb"
	 * @param axesOrder
	 * 	original axes order
	 * @return axes order with all the axes
	 */
	private static String convertToCompleteAxesOrder(String axesOrder) {
		String allDims = "xyczb";
		for (String ax : axesOrder.split(""))
			allDims = allDims.replace(ax, "");
		axesOrder = axesOrder + allDims;
		if (axesOrder.length() != 5)
			throw new IllegalArgumentException("the axesOrder parameter contains not allowed axes. The only"
					+ "allowe axes are: x, y, c, z and b");
		return axesOrder;
		
	}

}
