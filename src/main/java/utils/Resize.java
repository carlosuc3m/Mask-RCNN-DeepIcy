package utils;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Resize {
	
	public static void upscaleXY(INDArray im, int scaleX, int scaleY, String axesOrder) {
		String nAxesOrder = convertToCompleteAxesOrder(axesOrder);
		long[] shape = arrayToWantedAxesOrderAddOnes(im.shape(), axesOrder, nAxesOrder);
		int xInd = axesOrder.toLowerCase().indexOf("x");
		int yInd = axesOrder.toLowerCase().indexOf("y");
		int nX = scaleX * (int) shape[xInd];
		int nY = scaleY * (int) shape[yInd];
		
		
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

