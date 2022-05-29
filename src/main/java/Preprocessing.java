
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import utils.MaskRcnnMetas;
import utils.ImageProcessingUtils;
import utils.MaskRcnnAnchors;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;


public class Preprocessing {
	
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
	/**
	 * Parameters corresponding to the Mask R-CNN pre-processing
	 */
	private float[] WINDOW_SIZE;
	private float[] ORIGINAL_IMAGE_SIZE;
	private float[] PROCESSING_IMAGE_SIZE;
	private double SCALE;
	int NUM_CLASSES = 0;
	
	private Tensor inputTensor;
	
	private INDArray inputArray;
	
	private String configFileName;
	
	private String axesOrder;

	/**
	 * Constructor that adds the input image to the 
	 * @param inpTensor
	 */
	public Preprocessing(Tensor inpTensor) {
		inputTensor = inpTensor;
	}
	
	/**
	 * Constructor that adds the input image to the 
	 * @param inpTensor
	 */
	public Preprocessing(INDArray array) {
		inputArray = array;
	}
	
	public void setInputFileName(String name) {
		configFileName = name;
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
	 * This method replicates the Python pre-processing implemented at:
	 * https://github.com/matterport/Mask_RCNN/blob/3deaec5d902d16e1daf56b62d5971d428dc920bc/mrcnn/model.py#L2417
	 * 
	 * Method containing the whole Java pre-processing routine. 
	 * @param map: inputs to be pre-processed. It is provided by deepImageJ. The keys
	 * correspond to name given by the model to the inputs. And the values are the images
	 * selected to be applied to the model and any ResultsTable that is called as any of
	 * the parameter inputs of the model
	 * @return this method has to return HashMap whose keys are the inputs to the model as
	 * named by the model. The values types depend on the input type of tensor. For images,
	 * they should correspond to an ImagePlus. FOr parameters, the output provided should be either
	 * a Tensorflow tensor or a DJL NDArray
	 * Here is some documentation about creating Tensorflow tensors from Java Arrays:
	 * See <a href="https://www.tensorflow.org/api_docs/java/org/tensorflow/Tensors#public-static-tensorfloat-create-float[][][]-data">https://www.tensorflow.org/api_docs/java/org/tensorflow/Tensors#public-static-tensorfloat-create-float[][][]-data</a>
	 * 
	 * To create DJL NDArrays:
	 * See <a href="https://javadoc.io/doc/ai.djl/api/latest/ai/djl/ndarray/NDManager.html">https://javadoc.io/doc/ai.djl/api/latest/ai/djl/ndarray/NDManager.html</a>
	 */
    public HashMap<String, Object> apply() {
    	
    	if (inputTensor != null) {
    		inputArray = inputTensor.getData();
    	} else if (inputArray == null) {
    		throw new IllegalArgumentException("Runnning the pre-processing requires some input tensor.");
    	}
        
    	inputArray.castTo(DataType.FLOAT);

        // Create the dictionary of outputs that is going to be outpued by the pre-processing
        final HashMap<String, Object> map = new HashMap<String, Object>();
        // Create the ImagePlus that is going to result from pre-processing and apply the corresponding transformations
        //ImagePlus result = IJ.createImage(im.getTitle(), "32-bit", im.getWidth(), im.getHeight(), im.getNChannels(), im.getNSlices(), 1);
        moldInputs(inputArray);
        
        MaskRcnnAnchors mrccAnchors = new MaskRcnnAnchors(CONFIG);
        final float[][][] imageAnchors = MaskRcnnAnchors.getAnchors(inputArray, axesOrder);
        final Tensor anchors = Tensor.build("input_anchors", "brc", Nd4j.create(imageAnchors));
        anchors.setIsImage(false);
        
        //final float[][] imageMetas = MaskRcnnMetas.composeImageMeta(im);
        final float[][] imageMetas = MaskRcnnMetas.composeImageMeta(0.0f, ORIGINAL_IMAGE_SIZE, PROCESSING_IMAGE_SIZE, WINDOW_SIZE, (float) SCALE, NUM_CLASSES);
        final Tensor metas = Tensor.build("input_image_meta", "br", Nd4j.create(imageMetas));
        metas.setIsImage(false);
        
        // Write the runtime parameters to the config file so it can be used by post processing
        writeToConfigFile(CONFIG_FILE_PATH);
        
        // Create the output map
        map.put("input_image", inputArray);
        map.put("input_image_meta", metas);
        map.put("input_anchors", anchors);
        return map;
    }
    
    /**
	 * Auxiliary method to be able to change some pre-processing parameters without
	 * having to change the code. DeepImageJ gives the option of providing a extra
	 * files in the pre-processing which can be used for example as config files.
	 * It can act as a config file because the needed parameters can be specified in
	 * a comment block and the parsed by the pre-processing method
	 * @param configFiles: list of attachments. The files used by the pre-processing
	 * can then be selected by the name 
	 */
    public void setConfigFiles() {
    	String parentDir = new File(Preprocessing.class.getProtectionDomain().getCodeSource()
    			.getLocation().getFile()).getParent();
    	CONFIG_FILE_PATH = new File(parentDir, configFileName).getAbsolutePath();

    	if (CONFIG_FILE_PATH == null) {
    		ERROR = "No parameters file or config file provided for pre-processing.";
    		new IllegalArgumentException(ERROR);
    		return;
    	} else if (!(new File(CONFIG_FILE_PATH).exists())) {
    		ERROR = "The configuration file provided during pre-processing does not exist.";
    		new IllegalArgumentException(ERROR);
    		return;
    	}
    	// Parse parameters from the config file
    	// Parameters are saved in the HashMap 'config'
    	getParameters(CONFIG_FILE_PATH);
    }
    
    /**
     *  Takes an image and modifies it to the format expected by the
     *  neural network
     *  @param images: image to be modified
     *  @return the modified image ready to be processed
     *  
     */
    private void moldInputs(INDArray image) {
        // Get the parameters from the class atribute dictionary
    	int IMAGE_MIN_DIM = 0;
    	double IMAGE_MIN_SCALE = 0;
    	int IMAGE_MAX_DIM = 0;
    	String IMAGE_RESIZE_MODE = null;
    	
    	try {
        	IMAGE_MIN_DIM = Integer.parseInt(CONFIG.get("IMAGE_MIN_DIM"));
        	IMAGE_MIN_SCALE = (double) Float.parseFloat(CONFIG.get("IMAGE_MIN_SCALE"));
        	IMAGE_MAX_DIM = Integer.parseInt(CONFIG.get("IMAGE_MAX_DIM"));
        	IMAGE_RESIZE_MODE = CONFIG.get("IMAGE_RESIZE_MODE");
        	NUM_CLASSES = Integer.parseInt(CONFIG.get("NUM_CLASSES"));
    	} catch (Exception ex) {
    		ERROR = "Cannot parse correctly the parameters 'IMAGE_MIN_DIM', 'IMAGE_MIN_SCALE',\n"
    				+ "'IMAGE_MAX_DIM', 'IMAGE_RESIZE_MODE' and 'NUM_CLASSES' from the config file.";
    		throw new IllegalArgumentException(ERROR);
    	}
    	resizeImage(image, IMAGE_MIN_DIM, IMAGE_MIN_SCALE, IMAGE_MAX_DIM, IMAGE_RESIZE_MODE);
    	
    	long[] shape = image.shape();
    	final float[] finalShape = { (float)shape[axesOrder.indexOf("y")], (float)shape[axesOrder.indexOf("x")], (float)shape[axesOrder.indexOf("c")] };
    	moldImage(image, CONFIG);
    	
    	// Obtain the image meta data
    	PROCESSING_IMAGE_SIZE = finalShape;
    	
    }
    
    /**
     * This method subtracts the mean to each of the channels
     * @param moldedImage: image to modify
     * @param config: HashMap containing every parameter
     * @return modified image
     */
    private void moldImage(INDArray moldedImage, HashMap<String, String> config) {

		int cInd = axesOrder.indexOf("c");
		if (cInd == -1 || moldedImage.shape()[cInd] != 3) {
			throw new IllegalArgumentException("The input images should contain 3 channels. This one"
					+ " has " + moldedImage.shape()[cInd]);
		}
    	String MEAN_PIXEL_STRING;
    	try {
    		MEAN_PIXEL_STRING = config.get("MEAN_PIXEL");
    		MEAN_PIXEL_STRING = MEAN_PIXEL_STRING.substring(1, MEAN_PIXEL_STRING.length() - 1);
    		String[] aux = MEAN_PIXEL_STRING.split(",");
    		float[] MEAN_PIXEL = new float[aux.length];
    		for (int i = 0; i < aux.length; i ++) {
    			MEAN_PIXEL[i] = Float.parseFloat(aux[i]);
    		}
    		INDArrayIndex[] idxs = new INDArrayIndex[axesOrder.length()];
    		for (int i = 0; i < idxs.length; i ++) {
    			idxs[i] = NDArrayIndex.all();
    		}
    		for (int i = 0; i < MEAN_PIXEL.length; i ++) {
    			idxs[cInd] = NDArrayIndex.point(i);
    			moldedImage.put(idxs, moldedImage.get(idxs).add(-1 * MEAN_PIXEL[i]));
    		}
    	} catch (Exception ex) {
    		ERROR = "The config file information for the parameter 'MEAN_PIXEL' is incorrect or not present."
    				+ "\nThe value provided in the config file (" + CONFIG_FILE_PATH + "'\n"
    				+ "should be something like the following:\n"
    				+ " * RUNTIME_PARAMETER: WINDOW_SIZE = [112.0, 83.0, 912.0, 940.0]";
    		throw new IllegalArgumentException("Missing/Incorrect parameter: MEAN_PIXEL.\n" + ERROR);
    	}
    }
    
    /**
     * REsize the image keeping the aspect ratio unchanged
     * @param image: image to resize
     * @param minDim: resizes the image such that it's smaller
        dimension >= min_dim
     * @param maxDim: ensures that the image longest side doesn't
        exceed this value
     * @param minScale: ensures that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
     * @param mode: resizing mode
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.
     */
    private void resizeImage(INDArray image, int minDim, double minScale, int maxDim, 
    								String mode) {
    	// Default window is the whole image and default scale is 1
    	int w = (int) image.shape()[axesOrder.toLowerCase().indexOf("x")];
    	int h = (int) image.shape()[axesOrder.toLowerCase().indexOf("y")];
    	int c = (int) image.shape()[axesOrder.toLowerCase().indexOf("c")];
    	float[] window = new float[] {0, 0, h, w};
    	double scale = 1.0;
    	double[][] padding = new double[3][2];
    	
    	if (mode.equals("none")) {
    		SCALE = scale;
    		WINDOW_SIZE = window;
        	return;
    		
    	}
    	
    	// Find the scale
    	if (minDim > 0)
    		scale = Math.max(1.0, ((double) minDim) / Math.min(h,  w));
    	if (scale < minScale)
    		scale = minScale;
    	
    	// Check if the found scale exceeds the max dimension
    	double imageMax = Math.max(h, w);
    	if (mode.equals("square") && Math.round(imageMax * scale) > maxDim) {
    		scale = maxDim / imageMax;
    	}

        final float[] originalImShape = { (float) h, (float) w, (float) c };
    	// Obtain the image meta data
    	ORIGINAL_IMAGE_SIZE = originalImShape;
    	
    	if (scale != 1) {
    		ImageProcessingUtils.upscaleXY(image, (int)Math.round(w * scale), (int)Math.round(h * scale), axesOrder);
    	}
    	
    	// Check if padding is needed
    	if (mode.equals("square")) {
    		// Get the new h and w
        	w = (int) image.shape()[axesOrder.toLowerCase().indexOf("x")];
        	h = (int) image.shape()[axesOrder.toLowerCase().indexOf("y")];
        	c = (int) image.shape()[axesOrder.toLowerCase().indexOf("c")];
        	double topPad = Math.floor((maxDim - h) / 2.0);
        	double bottomPad = maxDim - h - topPad;
        	double leftPad = Math.floor((maxDim - w) / 2.0);
        	double rightPad = maxDim - w - leftPad;
        	padding[0][0] = topPad; padding[0][1] = bottomPad;
        	padding[1][0] = leftPad; padding[1][1] = rightPad;
        	ImageProcessingUtils.pad(image, padding, 0, axesOrder);
        	window = new float[] {(float) topPad, (float) leftPad, (float) (h + topPad), (float) (w + leftPad)};
    	} else if (mode.equals("pad64")) {
    		// Get the new h and w
        	w = (int) image.shape()[axesOrder.toLowerCase().indexOf("x")];
        	h = (int) image.shape()[axesOrder.toLowerCase().indexOf("y")];
        	c = (int) image.shape()[axesOrder.toLowerCase().indexOf("c")];
        	if (h % 64 != 0 || w % 64 != 0) {
        		ERROR = "The 'IMAGE_RESIZE_MODE = pad64' can only be applied to images\n"
        				+ "whose height and width are multiple of 64.";
        		throw new IllegalArgumentException(ERROR);
        	}
        	// Height
        	int topPad = 0;
        	int bottomPad = 0;
            if (h % 64 > 0) {
                int maxH = h - (h % 64) + 64;
                topPad = (maxH - h) / 0;
                bottomPad = maxH - h - topPad;
            }
        	// Width
        	int leftPad = 0;
        	int rightPad = 0;
            if (h % 64 > 0) {
                int maxW = w - (w % 64) + 64;
                leftPad = (maxW - w) / 2;
                rightPad = maxW - h - leftPad;
            }
        	padding[0][0] = topPad; padding[0][1] = bottomPad;
        	padding[1][0] = leftPad; padding[1][1] = rightPad;
        	ImageProcessingUtils.pad(image, padding, 0, axesOrder);
        	window = new float[] {(float) topPad, (float) leftPad, (float) (h + topPad), (float) (w + leftPad)};
    	} else if (mode.equals("crop")) {
    		ERROR = "This Java Mask R-CNN pre-processing does not support 'IMAGE_RESIZE_MODE = crop',\n"
    				+ "please change the parameter IMAGE_RESIZE_MODE to 'square' or 'pad64' in the\n"
    				+ "config pre-preprocessing file.";
    		throw new IllegalArgumentException(ERROR);
    	} else {
    		ERROR = "The config file information for the parameter 'IMAGE_RESIZE_MODE' is incorrect."
    				+ "\nThe value provided in the config file (" + CONFIG_FILE_PATH + "')\n"
					+ "is '" + mode + "'. However the only values allowed are: 'square', 'pad64' and 'XXX'.";
    		throw new IllegalArgumentException(ERROR);
    	}
    	
    	// Set the class attributes to be used later
    	WINDOW_SIZE = window;
    	SCALE = scale;

    	return;
    }
    
    /**
     * Parse parameters from a file provided in the plugin.
     * This method will try to find if there is either a .ijm or .txt file provided for
     * post-processing and if there exists, it will be inputted to this method.
     * Inside this method now we can do whatever we want to the file to extract the wanted parameters.
     * In this particular case, the method will parse the text and look for the string: '* PARAMETER:' 
     * the corresponding parameter key will be right after, the '=' and finally the parameter value just before
     * the new line string '\n'
     * @param parametersFile: file containing parameters needed for post-processing provided in the plugin
     */
    private void getParameters(String parametersFile) {
    	// Initialise the parameters dictionary
    	CONFIG = new HashMap<String, String>();
    	// For this particular case, because the program is going to later
    	// modify the plugin, save the path to the file in an attribute
    	CONFIG_FILE_PATH = parametersFile;
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
     * Writes runtime parameters needed during post processing into the config file
     * @param path: path to the config file
     */
    private void writeToConfigFile(String path) {
    	String finalStr = "";
    	String runtimeParametersSectionFlag = "- PARAMETERS_MODIFIED_AT_RUNTIME -";
    	String runtimeParameterFlag = "RUNTIME_PARAMETER:";
    	String separator = "=";
    	try (BufferedReader br = new BufferedReader(new FileReader(path))) {
    	    String line = br.readLine();
    	    boolean now = false;
    	    while (line != null) {
    	       if (line.contains(runtimeParametersSectionFlag) && !line.contains("'" + runtimeParametersSectionFlag + "'")) {
    	    	   now = true;
    	       }
    	       if (now && line.contains(runtimeParameterFlag)) {
    	    	   line = line.trim();
    	    	   int paramStart = line.indexOf(runtimeParameterFlag) + runtimeParameterFlag.length();
    	    	   int separatorInd = line.indexOf(separator);
    	    	   // Parameter key and value are separated by '='
    	    	   String key = line.substring(paramStart, separatorInd).trim();
    	    	   if (key.contentEquals("WINDOW_SIZE"))
    	    		   line = " * RUNTIME_PARAMETER: WINDOW_SIZE = " + Arrays.toString(WINDOW_SIZE);
    	    	   else if (key.contentEquals("ORIGINAL_IMAGE_SIZE"))
    	    		   line = " * RUNTIME_PARAMETER: ORIGINAL_IMAGE_SIZE = " + Arrays.toString(ORIGINAL_IMAGE_SIZE);
    	    	   else if (key.contentEquals("PROCESSING_IMAGE_SIZE"))
    	    		   line = " * RUNTIME_PARAMETER: PROCESSING_IMAGE_SIZE = " + Arrays.toString(PROCESSING_IMAGE_SIZE);
    	       }
    	       finalStr += line + System.getProperty("line.separator");
    	       line = br.readLine();
    	    }
    	    br.close();
			// Write the file
			BufferedWriter out = new BufferedWriter(new FileWriter(path));
			out.write(finalStr);
			out.close();
    	} catch (IOException e) {
			ERROR = "Cannot find pre-processing config file (" + CONFIG_FILE_PATH + ").\n" +
					"Runtime parameters cannot be overwritten, post-processing might fail.";
			e.printStackTrace();
		}
    }
}
