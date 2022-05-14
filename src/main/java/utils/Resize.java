package utils;

import java.util.stream.IntStream;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Resize {
	
	public static void upscaleXY(INDArray im, int scaleX, int scaleY, String axesOrder) {
		String nAxesOrder = convertToCompleteAxesOrder(axesOrder);
		long[] shape = arrayToWantedAxesOrderAddOnes(im.shape(), axesOrder, nAxesOrder);
		int xInd = nAxesOrder.toLowerCase().indexOf("x");
		int yInd = nAxesOrder.toLowerCase().indexOf("y");
		int nX = scaleX * (int) shape[xInd];
		int nY = scaleY * (int) shape[yInd];
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
									getFlatPosOfNDArray(position, shape)
								}
							}
						}
					}
				}
			}
		}
		
		
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

