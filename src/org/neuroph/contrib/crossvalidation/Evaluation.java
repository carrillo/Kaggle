package org.neuroph.contrib.crossvalidation;

import inputOutput.TextFileAccess;

import java.io.PrintWriter;
import java.util.ArrayList;

import org.neuroph.core.learning.error.ErrorFunction;

import array.tools.DoubleArrayTools;

public class Evaluation 
{
	private ArrayList<double[]> prediction; 
	private ArrayList<double[]> observation; 
	
	
	public Evaluation( final ArrayList<double[]> prediction, final ArrayList<double[]> observation ) {
		this.prediction = prediction; 
		this.observation = observation; 
	}
	
	/**
	 * Calculate mean error with a defined error function 
	 * @param errorfunction
	 */
	public double getMeanError( final ErrorFunction errorfunction ) {
		
		errorfunction.reset(); 
		final double[] diff = getOutputDifference(); 
		errorfunction.addOutputError( diff );
		return errorfunction.getTotalError(); 
	}
	
	/**
	 * Calculate difference between predicted and deserved output 
	 * @return
	 */
	private double[] getOutputDifference() {
		
		final int outputSize = prediction.get( 0 ).length; 
		
		final double[] diff = new double[ prediction.size() * outputSize ]; 
		
		int diffIndex = 0;  
		for ( int i = 0; i < prediction.size(); i++ ) {
			for( int j = 0; j < outputSize; j++ ) {
				diff[ diffIndex ] = prediction.get( i )[ j ] - observation.get( i )[ j ];
				diffIndex++; 
			}
        }
		
		return diff; 
	}
	
	/**
	 * Write to file
	 * @param fileName
	 */
	public void write( final String fileName ) {
		PrintWriter out = TextFileAccess.openFileWrite( fileName ); 
		
		//out.println( "observed" + "," + "predicted" ); 
		
		for( int i = 0; i < prediction.size(); i++ ) {
			
			String obs = DoubleArrayTools.arrayToString( observation.get( i ), ",");
			String pred = DoubleArrayTools.arrayToString( prediction.get( i ), ",");
			
			out.println( obs + "\t" + pred );  
		}
		
		out.flush();
		out.close();
	}
	
	//Getter 
	public ArrayList<double[]> getPrediction() { return prediction; }
	public ArrayList<double[]> getObservation() { return observation; }
}
