package org.neuroph.contrib.crossvalidation;

import inputOutput.TextFileAccess;

import java.io.PrintWriter;
import java.util.ArrayList;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.core.transfer.TransferFunction;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.util.TransferFunctionType;

import array.tools.DoubleArrayTools;

import com.sun.tools.javac.util.List;

public class HiddenNodeLearningRule 
{
	private DataSet data; 
	private TransferFunctionType transferFunctionType; 
	private int foldCrossValidation; 
	private int min, max; 
	private PrintWriter log; 
	
	private NeuralNetwork ann; 
	private LearningRule learningRule; 
	
	
	private ArrayList<Integer> neuronsInLayer = new ArrayList<Integer>(); 
	
	public HiddenNodeLearningRule( final DataSet data, final int foldCrossValidation,  
			final TransferFunctionType transferFunctionType, final LearningRule learningRule, 
			final String logFile ) {
		
		this.data = data; 
		this.foldCrossValidation = foldCrossValidation; 
		
		this.transferFunctionType = transferFunctionType;
		this.learningRule = learningRule; 
		
		this.log = TextFileAccess.openFileWrite( logFile ); 
	}
	
	public void run( final int minHiddenNodes, final int maxHiddenNodes, final int step ) {
		
		//Set up the neurons in layer list 
		neuronsInLayer.add( data.getInputSize() ); 
		neuronsInLayer.add( minHiddenNodes ); 
		neuronsInLayer.add( data.getOutputSize() ); 
		
		
		for( int i = minHiddenNodes; i <= maxHiddenNodes; i += step ) {
			
			neuronsInLayer.set( 1, i );  
			
			ann = new MultiLayerPerceptron( neuronsInLayer, this.transferFunctionType );
			ann.setLearningRule( this.learningRule );
			
			CrossValidation cv = new CrossValidation( this.data, ann, foldCrossValidation ); 
			cv.run(); 
			this.log.println( i + "," + DoubleArrayTools.arrayToString( cv.getError(), "," ) );
			this.log.flush(); 
			
			System.out.println( "Hidden nodes: " + i + "\t\tMeanError: " +  cv.getMeanError() ); 
		}
		
		this.log.close(); 
		
	}
}
