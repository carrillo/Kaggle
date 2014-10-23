package org.neuroph.contrib.crossvalidation;

import inputOutput.TextFileAccess;

import java.io.PrintWriter;
import java.util.ArrayList;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

import array.tools.DoubleArrayTools;

public class CrossValidation 
{
	private DataSet[] foldedData; 
	private NeuralNetwork ann; 
	
	private ArrayList<Evaluation> evaluations = new ArrayList<Evaluation>(); 
	
	public CrossValidation( final DataSet data, final NeuralNetwork ann, final int kFold ) {
		//Fold data
		this.foldedData = data.sample( new FoldData( kFold ) ); 
		this.ann = ann;  
	}
	
	
	/**
	 * 1. Split data into k folds. 
	 * 2. Run training and evaluations on folded sets. 
	 */
	public void run() {
		
		DataSet test = null;
		DataSet train = null;
		
		//Perform training and evaluation for all combinations of training and test. 
		for( int i = 0; i < foldedData.length; i++ ) {			
			
			test = foldedData[ i ];  
			train = getTrainingSet( i ); 
			
			//System.out.println( test.getRows().size() + "\t" + train.getRows().size() );
			
			ann.randomizeWeights(); 
			ann.reset();
			ann.learn( train );
	
			Evaluation eval = evaluate( test ); 
			
			//System.out.println( eval.getMeanError( new MeanSquaredError( test.size() ) ) ); 

			evaluations.add( eval ); 	
		}
	}
	
	/**
	 * Returns error over cross validation sets. 
	 * @return
	 */
	public double[] getError() {
		final double[] errors = new double[ evaluations.size() ]; 
		
		Evaluation eval; 
		for( int i = 0; i < errors.length; i++ ) {
			eval = evaluations.get( i ); 
			errors[ i ] = eval.getMeanError( new MeanError( eval.getObservation().size() ) );
		}
		
		return errors; 
	}
	
	public double getMeanError() {
		DescriptiveStatistics ds = new DescriptiveStatistics( getError() );  
		return ds.getMean(); 
	}
 	
	public void writeObservedVsPredictedSingleFiles( final String fileNameLeader ) {
		
		String fileName; 
		for( int i = 0; i < evaluations.size(); i++ ) {
			
			fileName = "xValidationObsVsPredicted" + i + ".csv"; 
			
			evaluations.get( i ).write( fileNameLeader + fileName );
		}
		
	}
	
	/**
	 * Write observed and predicted values for all cv runs. 
	 * @param fileName
	 */
	public void writeAllObservedVsPredictedPairs( final String fileName ) {
		
		PrintWriter out = TextFileAccess.openFileWrite( fileName ); 
		
		final ArrayList<double[]> observationCollect = new ArrayList<double[]>(); 
		final ArrayList<double[]> predictionCollect = new ArrayList<double[]>();
	
		
		for( Evaluation eval : evaluations ) {
			observationCollect.addAll( eval.getObservation() ); 
			predictionCollect.addAll( eval.getPrediction() ); 
			 
		}
		
		
		out.println( "observation" + "\t" + "prediction" ); 
		for( int i = 0; i < predictionCollect.size(); i++ ) {
			
			String obs = DoubleArrayTools.arrayToString( observationCollect .get( i ), ",");
			String pred = DoubleArrayTools.arrayToString( predictionCollect.get( i ), ",");
			
			out.println( obs + "\t" + pred );  
		}
		
		out.flush();
		out.close();
		
	}
	
	/**
	 * Write observed and predicted values for all cv runs. 
	 * @param fileName
	 */
	public void writeAllMeanErrors( final String fileName ) {
		
		PrintWriter out = TextFileAccess.openFileWrite( fileName ); 
		
		for( Evaluation eval : evaluations ) {
				
			out.println( eval.getMeanError( new MeanError( eval.getObservation().size() )) );  
		}
		
		out.flush();
		out.close();
		
	}
	
	
	private Evaluation evaluate( final DataSet test ) {
		
		ArrayList<double[]> prediction = new ArrayList<double[]>();
		ArrayList<double[]> observation = new ArrayList<double[]>();
		
		for( DataSetRow dataRow : test.getRows() )
		{
			ann.setInput( dataRow.getInput() );
			ann.calculate(); 
			prediction.add( ann.getOutput().clone() ); 
			observation.add( dataRow.getDesiredOutput().clone() ); 
			  
		}
		
		
		return new Evaluation( prediction, observation ); 
	}
	
	/**
	 * Combine all except the test-set
	 * @param currentFold
	 * @return
	 */
	private DataSet getTrainingSet( final int currentFold ) {
		
		final int inputSize = foldedData[ 0 ].getInputSize(); 
		final int outputSize = foldedData[ 0 ].getOutputSize();
		
		final DataSet out = new DataSet(inputSize, outputSize); 
		
		for( int i = 0; i < foldedData.length; i++ ) {
			if( i == currentFold ) {
				continue; 
			}
			
			addAllRows( foldedData[ i ], out ); 
		}
		
		return out; 
	}
	
	/**
	 * Adds all rows from DataSet1 to DataSet2
	 * @param dataSet1
	 * @param dataSet2
	 */
	private void addAllRows( final DataSet dataSet1, final DataSet dataSet2 ) {
		for( int i = 0; i < dataSet1.getRows().size(); i++ ) {
			dataSet2.addRow( dataSet1.getRowAt( i ) );
		}
	}
	
	
}
