package wekaTools;

import java.io.File;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

/**
 * This class converts csv files to Arff files used in WEKA. 
 * I copied and modified this class from "http://weka.wikispaces.com/Converting+CSV+to+ARFF" 
 * to get familiar with WEKA  
 *
 */
public class CSV2Arff {

	/**
	   * takes 2 arguments:
	   * - CSV input file
	   * - ARFF output file
	   */
	public static void main(String[] args) throws IOException 
	{
		/*
		if (args.length != 2) 
		{
		      System.out.println("\nUsage: CSV2Arff <input.csv> <output.arff>\n");
		      System.exit(1);
		}
		*/
		
		final String in = "/Users/carrillo/workspace/Kaggle/resources/titanic/a.csv"; 
		final String out = "/Users/carrillo/workspace/Kaggle/resources/titanic/trainClean.arff";
		
		// load CSV
	    CSVLoader loader = new CSVLoader();
	    loader.setSource( new File( in ) );
	    Instances data = loader.getDataSet();
	    
	    // save ARFF
	    ArffSaver saver = new ArffSaver();
	    saver.setInstances(data);
	    saver.setFile(new File( out ) );
	    //saver.setDestination(new File(args[1]));
	    saver.writeBatch();

	}

}
