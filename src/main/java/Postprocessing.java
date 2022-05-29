import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import utils.ImageProcessingUtils;

public class Postprocessing {
	/**
	 * Dictionary containing all the parameters parsed from the file
	 */
	private static HashMap<String, String> CONFIG = new HashMap<String, String>();
	/**
	 * Path to the other pre-processing file provided in deepImageJ. If it contains 
	 * either a .ijm or .txt file it will be parsed to find parameters
	 */
	private static String CONFIG_FILE_PATH;
	/**
	 * Attribute to communicate errors to DeepImageJ plugins
	 */
	private static String ERROR = "";
	
	private Tensor inputTensor;
	
	private Tensor detectionTensor;
	
	private String configFileName;
	
	private String axesOrder;

	/**
	 * Constructor that adds the input image to the 
	 * @param inpTensor
	 */
	public Postprocessing(Tensor inpTensor) {
		inputTensor = inpTensor;
	}
	
	public void setInputFileName(String name) {
		configFileName = name;
	}
	
	public void setSetectionTensor(Tensor tt) {
		detectionTensor = tt;
	}
	
	public void setAxesOrder(String str) {
		axesOrder = str;
	}

	/**
	 * Return error that stopped pre-processing to DeepImageJ
	 */
	public String error() {
		return ERROR;
	}
	
	/**
	 * This method does the equivalent to unmold_detections at:
	 * https://github.com/matterport/Mask_RCNN/blob/3deaec5d902d16e1daf56b62d5971d428dc920bc/mrcnn/model.py#L2417
	 * 
	 * 
	 * Method containing the whole Java post-processing routine. 
	 * @param map: outputs to be post-processed. It is provided by deepImageJ. The keys
	 * correspond to name given by the model to the outputs. And the values are the images and 
	 * ResultsTables outputes by the model.
	 * @return this method has to return a HashMap with the post-processing results.
	 */
    public HashMap<String, Tensor> apply() {
        // Get the number of objects detected by the net
        final int nDetections = getNDetections();
        // If nothing was detected just return null
        if (nDetections == 0) {
        	ERROR = "No object was detected in the input image.";
        	return null;
        }
        // Get the detected bounding boxes from the output table in normalised coodinates
        final double[][] boxes = new double[4][nDetections];
        // Get the class IDs of the detected objects
        int[] classIds = new int[nDetections];
        boxes[0] = detectionTensor.getData().get(NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all()).toDoubleVector();
        boxes[1] = detectionTensor.getData().get(NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.all()).toDoubleVector();
        boxes[2] = detectionTensor.getData().get(NDArrayIndex.all(), NDArrayIndex.point(2), NDArrayIndex.all()).toDoubleVector();
        boxes[3] = detectionTensor.getData().get(NDArrayIndex.all(), NDArrayIndex.point(3), NDArrayIndex.all()).toDoubleVector();
        classIds = detectionTensor.getData().get(NDArrayIndex.all(), NDArrayIndex.point(4), NDArrayIndex.all()).toIntVector();
        // Select the masks corresponding to the objects detected
        INDArray selectedMasks = Nd4j.zeros(new int[] {detectionTensor.getShape()[0], detectionTensor.getShape()[1], detectionTensor.getShape()[2], nDetections});
        int z = 0;
        int[] array;
        INDArray mask;
		for (int length = (array = classIds).length, l = 0; l < length; l ++) {
            final int classId = array[l];
            mask = detectionTensor.getData().get(NDArrayIndex.all(), NDArrayIndex.point(l), NDArrayIndex.all(),
            		NDArrayIndex.all(), NDArrayIndex.point(classId));
            selectedMasks.put(new INDArrayIndex[] {(INDArrayIndex) NDArrayIndex.all(), (INDArrayIndex) NDArrayIndex.all(),
            		(INDArrayIndex) NDArrayIndex.all(), (INDArrayIndex) NDArrayIndex.point(l)}, mask);
        }

        // String get the needed parameters from the config file
        String originalShapeString = CONFIG.get("ORIGINAL_IMAGE_SIZE");
        String processingShapeString = CONFIG.get("PROCESSING_IMAGE_SIZE");
        String windowString = CONFIG.get("WINDOW_SIZE");
        // Get an float arrays from the strings
        float[] originalShape = str2array(originalShapeString);
        float[] processingShape = str2array(processingShapeString);
        float[] window = str2array(windowString);
        
        INDArray finalMasks = Nd4j.zeros(detectionTensor.getShape()[0], (int) Math.floor(originalShape[0]), (int) Math.floor(originalShape[1]), nDetections);
        
        // Denormalize the bounding boxes to pixel coordinates in the processing shape
        window = normBoxes(window, processingShape);
        float[] shift = {window[0], window[1], window[0], window[1]};
        // Window height
        float wh = window[2] - window[0];
        // Window width
        float ww = window[3] - window[1];
        float[] scale = {wh, ww, wh, ww};
        // Convert boxes to pixel coordinates of the original image
        for (int i = 0; i < boxes.length; i ++) {
        	for (int j = 0; j < boxes[0].length; j ++) {
        		boxes[i][j] = (boxes[i][j] - shift[j]) / scale[j];
        	}
        }
        
        // Get the final boxes that indicate where is the mask located in the image
        final int[][] scaledBoxes = denormBoxes(boxes, originalShape);
        // Paste the mask into their corresponding places
        for (int j = 0; j < classIds.length; ++j) {
            int newHeight = scaledBoxes[j][2] - scaledBoxes[j][0];
            int newWidth = scaledBoxes[j][3] - scaledBoxes[j][1];
            ImageProcessingUtils.upscaleXY(selectedMasks.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
            		NDArrayIndex.point(j)), newWidth / (int) selectedMasks.shape()[2], newHeight / (int) selectedMasks.shape()[1], "byxc");
    		INDArrayIndex[] idxs = new INDArrayIndex[4];
    		idxs[0] = NDArrayIndex.all();
    		idxs[1] = NDArrayIndex.interval(scaledBoxes[j][0], scaledBoxes[j][2]);
    		idxs[2] = NDArrayIndex.interval(scaledBoxes[j][1], scaledBoxes[j][3]);
    		idxs[3] = NDArrayIndex.all();
    		
            finalMasks.put(idxs, selectedMasks.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(j)));
            // TODO add threshold for 0.5
        }
        selectedMasks.close();
        inputTensor.close();
        final HashMap<String, Tensor> outMap = new HashMap<String, Tensor>();
        outMap.put("detections", detectionTensor);
        outMap.put("mask", Tensor.build("masks", "byxc", finalMasks));
        return outMap;
    }

    /**
	 * Auxiliary method to be able to change some post-processing parameters without
	 * having to change the code. DeepImageJ gives the option of providing a extra
	 * files in the post-processing which can be used for example as config files.
	 * It can act as a config file because the needed parameters can be specified in
	 * a comment block and the parsed by the post-processing method
	 * @param configFiles: list of attachments. The files used by the post-processing
	 * can then be selected by the name 
	 */
    public void setConfigFiles(ArrayList<String> configFiles) {
	    	for (String ff : configFiles) {
	    		String fileName = ff.substring(ff.lastIndexOf(File.separator) + 1);
	    		if (fileName.contentEquals("config.ijm")) {
	    	    	CONFIG_FILE_PATH = ff;
	    	    	break;
	    		}
	    	}
	    	if (CONFIG_FILE_PATH == null && configFiles.size() == 0) {
	    		ERROR = "No parameters file or config file provided for post-processing.";
	    		return;
	    	} else if (CONFIG_FILE_PATH == null && configFiles.size() > 0) {
	    		ERROR = "A configuration file was not found in the model. The configuration file"
	    				+ "should be called 'config.ijm', please rename the config file if it is "
	    				+ "not named correctly.";
	    		return;
	    	} else if (!(new File(CONFIG_FILE_PATH).exists())) {
	    		ERROR = "The configuration file provided during post-processing does not exist.";
	    		return;
	    	}
	    	// Parse parameters from the config file
	    	// Parameters are saved in the HashMap 'config'
	    	getParameters(CONFIG_FILE_PATH);
	    }
    
    /**
     * Parse parameters from a file provided in the plugin.
     * This method will try to find if there is either a .ijm or .txt file provided for
     * post-processing and if there exists, it will be inputed to this method.
     * Inside this method now we can do whatever we want to the file to extract the wanted parameters.
     * In this particular case, the method will parse the text and look for the string: '* PARAMETER:' 
     * the corresponding parameter key will be right after, the '=' and finally the parameter value just before
     * the new line string '\n'
     * @param parametersFile: file containing parameters needed for post-processing provided in the plugin
     */
    public void getParameters(String parametersFile) {
    	// Initialise the parameters dictionary
    	CONFIG = new HashMap<String, String>();
    	File configFile = new File(parametersFile);
    	// Key that is used to know where is each parameter
    	String flag = "PARAMETER:";
    	String flag2 = "*";
    	String separator = "=";
    	// Read the file line by line
    	try (BufferedReader br = new BufferedReader(new FileReader(configFile))) {
    	    String line = br.readLine().trim();
    	    while (line != null) {
    	    	line = line.trim();
    	       if (line.contains(flag) && line.contains(flag2) && !line.contains("'" + flag + "'")) {
    	    	   int paramStart = line.indexOf(flag) + flag.length();
    	    	   int separatorInd = line.indexOf(separator);
    	    	   // Parameter key and value are separated by '='
    	    	   String key = line.substring(paramStart, separatorInd).trim();
    	    	   String value = line.substring(separatorInd + 1).trim();
    	    	   CONFIG.put(key, value);
    	       }
    	       line = br.readLine();
    	    }
    	    br.close();
    	} catch (IOException e) {
			ERROR = "Could not access the config file provided during pre-preocessing:\n"
					+ "- " + parametersFile;
			e.printStackTrace();
			CONFIG = null;
		}
    }
    
    /**
     * Get the number of objects detected by the model, that is rows that are non-zero
     * @return number of objects detected
     */
    private int getNDetections() {
        int n = 0;
        for (int i = 0; i < detectionTensor.getShape()[1]; ++i) {
            int label = detectionTensor.getData().getInt(new int[] {0, 4, i});
            if (label != 0) {
                n ++;
            } else {
            	break;
            }
        }
        return n;
    }
    
    /**
     * Converts boxes from pixel coordinates to normalized coordinates.
     * Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
     * coordinates it's inside the box
     * @param window: array containing the coordinates of the window that corresponds to
     * the original image, [top, left, bottom, right], i.e: [ y1, x1, y2, x2 ]
     * @param imageShape: array with height and width of the modified image 
     * @return box normalized coordinates as [y1, x1, y2, x2]
     */
    private static float[] normBoxes(float[] window, float[] imageShape) {
    	float h = imageShape[0];
    	float w = imageShape[1];
    	float[] scale = {h - 1, w - 1, h - 1, w - 1}; 
    	float[] shift = {0, 0, 1, 1}; 
    	float[] normBox = new float[scale.length];
    	for (int i = 0; i < normBox.length; i ++)
    		normBox[i] = (window[i] - shift[i]) / scale[i];
    	return normBox;
    }
    
    /**
     * Converts boxes from normalized coordinates to pixel coordinates.
     * Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
     * coordinates it's inside the box.
     * @param boxes: vertices for the bounding boxes of each of the
     * detected objects. In normalised coordinates
     * @param shape: height and width in pixels of an image
     * @return an array containing the vertices of each bounding
     * box in pixel coordinates
     */
    private static int[][] denormBoxes(final double[][] boxes, final float[] shape) {
        final float h = shape[0];
        final float w = shape[1];
        final double[] scale = { h - 1, w - 1, h - 1, w - 1 };
        final double[] shift = { 0.0, 0.0, 1.0, 1.0 };
        final int[][] newBoxes = new int[boxes.length][boxes[0].length];
        for (int i = 0; i < boxes.length; ++i) {
            newBoxes[i][0] = (int)Math.round(boxes[i][0] * scale[0] + shift[0]);
            newBoxes[i][1] = (int)Math.round(boxes[i][1] * scale[1] + shift[1]);
            newBoxes[i][2] = (int)Math.round(boxes[i][2] * scale[2] + shift[2]);
            newBoxes[i][3] = (int)Math.round(boxes[i][3] * scale[3] + shift[3]);
        }
        return newBoxes;
    }
    
    /**
     * Converts an array of the form '[a,b,c,d]' into a float array
     * @param str: string representation of an array
     * @return float array or null in the case it was not possible
     */
    public static float[] str2array(String str) {
    	try {
	    	if (str.indexOf("[") != -1)
	    		str = str.substring(str.indexOf("[") + 1);
	    	else if (str.indexOf("(") != -1)
	    		str = str.substring(str.indexOf("(") + 1);
	
	    	if (str.indexOf("]") != -1)
	    		str = str.substring(0, str.indexOf("]"));
	    	else if (str.indexOf(")") != -1)
	    		str = str.substring(0, str.indexOf(")"));
	    	
	    	String[] strArr = str.split(",");
	    	float[] arr = new float[strArr.length];
	    	for (int i = 0; i < strArr.length; i ++) {
	    		arr[i] = Float.parseFloat(strArr[i]);
	    	}
	    	return arr;
    	} catch (Exception ex){
    		return null;
    	}
    }
}
