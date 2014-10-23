package neurophTools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import org.neuroph.contrib.crossvalidation.CrossValidation;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.util.benchmark.MyBenchmarkTask;

import array.tools.ArrayListDoubleTools;
import array.tools.DoubleArrayTools;
import weka.core.Instances;
import wekaTools.WekaToNeurophTools;

public class GreedyFeatureSelection 
{
	private BackPropagation learningRule; 
	private Instances trainingSet; 
	
	private ArrayList<ArrayList<Double>> desiredOutput;
	private ArrayList<ArrayList<Double>> input;
	
	private ArrayList<String> featureLabels; 
	
	private HashMap<Integer, Integer> indexRankHashMap = new HashMap<Integer, Integer>();
	private HashMap<Integer, Double> indexWeightHashMap = new HashMap<Integer, Double>();
	
	public GreedyFeatureSelection( final BackPropagation learningRule, final Instances trainingSet ) throws Exception {
		this.learningRule = learningRule; 
		this.trainingSet = trainingSet; 
		
		initiateOutput();
		initiateInput(); 
	}
	
	/**
	 * Set the output values from the training data
	 * @throws Exception
	 */
	private void initiateOutput() throws Exception {
		ArrayList<ArrayList<Double>> output = new ArrayList<ArrayList<Double>>(); 
		
		ArrayList<double[]> temp = WekaToNeurophTools.getOuput( this.trainingSet );
		for( double[] d : temp ) {
			output.add( DoubleArrayTools.toArrayList( d ) );  
		}
		
		this.desiredOutput = output; 
	}
	
	/**
	 * Initiate the input selection with empty arraylists. 
	 */
	private void initiateInput() { 
		
		ArrayList<ArrayList<Double>> input = new ArrayList<ArrayList<Double>>(); 
		for( int i =0; i <  trainingSet.numInstances(); i++ ) {
			input.add( new ArrayList<Double>() ); 
		}
		this.input = input;
	}
	
	public void run() {
		
		for( int n = 0; n < trainingSet.numAttributes(); n++ ) {
			System.out.print( n + "\t" ); 
			final int indexToAdd = findNextBestFeature();
			addFeature( indexToAdd ); 
			indexRankHashMap.put( indexToAdd, 0 );
		}
		 
		 
		
	}
	
	/**
	 * Identify next best feature. 
	 * @return
	 */
	private int findNextBestFeature() {
		double meanError = Double.MAX_VALUE; 
		int index = -1; ;
		double currentError = -1; 
		for( int i = 0; i < trainingSet.numAttributes(); i++ ) {
			
			if( !this.indexRankHashMap.containsKey( i ) && i != trainingSet.classIndex() ) {
				
				currentError =  getEvaluation( i );
				
				if( currentError < meanError ) {
					meanError = currentError; 
					index = i; 
				}
			}
			
		}
		System.out.println( currentError + "\t" + index + "\t" + trainingSet.attribute( index ).name() );
		
		indexWeightHashMap.put( index, currentError ); 
		
		return index; 
	}
	
	private void addFeature( final int indexToAdd ) {
		
		for( int i = 0; i < trainingSet.numInstances(); i++ ) {
			input.get( i ).add( this.trainingSet.get( i ).value( indexToAdd ) ); 
		}
		
	}
	
	private Double getEvaluation( final int index ) { 
		
		DataSet dataSet = getNewTestData( index ); 
		
		//System.out.print( dataSet.getInputSize() + "\t" ); 
		
		final MultiLayerPerceptron ann = new MultiLayerPerceptron( TransferFunctionType.LINEAR, dataSet.getInputSize(), 1, dataSet.getOutputSize() ) ;
		ann.setLearningRule( this.learningRule );
		
		CrossValidation cv = new CrossValidation( dataSet, ann, 100 );
		cv.run();
		
		return cv.getMeanError(); 
		
	}
	
	private DataSet getNewTestData( final int indexToAdd ) {
		
		DataSet newTestData = new DataSet( ( this.input.get( 0 ).size() + 1 ), this.desiredOutput.get( 0 ).size() );
		
		ArrayList<Double> row; 
		for( int i = 0; i < this.input.size(); i++ ) {
			row = ( ArrayList<Double> ) this.input.get( i ).clone();
			row.add( this.trainingSet.get( i ).value( indexToAdd ) ); 
			
			newTestData.addRow( new DataSetRow( row, this.desiredOutput.get( i ) ) ); 
		}
		
		return newTestData; 
	}
	
	
}
