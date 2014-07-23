package neurophTools;

import java.io.PrintStream;
import java.io.PrintWriter;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.learning.MomentumBackpropagation;

import com.sun.java.swing.plaf.windows.WindowsBorders.DashedBorder;

import sun.reflect.ReflectionFactory.GetReflectionFactoryAction;
import sun.security.action.GetLongAction;

public class LearningCurve 
{
	private NeuralNetwork ann; 
	private Evaluation evaluation;
	private LearningRule learningRule; 
	
	public LearningCurve( NeuralNetwork ann )
	{
		this( ann, new MeanSquaredErrorHalf() ); 
	}
	public LearningCurve(  NeuralNetwork ann , DistanceMeasure errorFunction )
	{
		this( ann, errorFunction, standardLearningRule( 10, true ) );  
	}
	public LearningCurve(  NeuralNetwork ann , DistanceMeasure errorFunction, LearningRule learningRule )
	{
		this.ann = ann; 
		this.evaluation = new Evaluation( null, errorFunction ); 
		this.learningRule = learningRule; 
	}
	
	private static LearningRule standardLearningRule( final int iterations, final boolean verbose ) 
	{
		MomentumBackpropagation momentumBackpropagationLearningRule = new MomentumBackpropagation();
		momentumBackpropagationLearningRule.setLearningRate( 0.005 );
		momentumBackpropagationLearningRule.setMomentum( 0.00 );
		momentumBackpropagationLearningRule.setMaxIterations( iterations );
		momentumBackpropagationLearningRule.setMaxError( 0.0000000001 );
		if( verbose )
			momentumBackpropagationLearningRule.addListener( new SimpleLearningEventListener() );
		
		return momentumBackpropagationLearningRule; 
	}
	
	/**
	 * Generate a learning curve, determining the training set and test set error with increasing training set.  
	 * @param data
	 * @param ann
	 * @param maxDataFraction
	 * @param out
	 */
	public void writeLearningCurve( final DataSet data, final double maxDataFraction, 
			final PrintStream out )
	{
		if( maxDataFraction > 0.5 )
		{
			System.err.println( "MaxDataFraction larger than 0.5. Cannot generate non-overlapping training and validation data." );
			System.exit( 1 );
		}
		 int sampleCount = 2;
		 int maxSampleCount = (int) Math.floor( data.getRows().size() * maxDataFraction ); 
		 
		 while( sampleCount <= maxSampleCount )
		 {
			 System.out.println( sampleCount ); 
			 //Sample data to retrieve training and validation data 
			 final DataSet[] dataSets = getTrainingAndValidateData( data, sampleCount ); 
			 
			 //Learn Network
			 learnNetwork( dataSets[ 0 ] );
			 
			 //Report error 
			 reportError( dataSets, out );
			 
			 sampleCount = sampleCount * 2; 
		 }
		 
	}
	
	/**
	 * Draw 2*sampleCount samples from data and distribute to training and validation datasets.  
	 * @param data
	 * @param sampleCount
	 * @return
	 */
	private DataSet[] getTrainingAndValidateData( final DataSet data, final int sampleCount ) 
	{
		final DataSet sample = Sampling.subsampleWithoutReplacement( data, ( sampleCount * 2  ) ); 
		final DataSet[] out = new DataSet[]{ new DataSet( data.getInputSize() ), new DataSet( data.getInputSize() ) };
		
		for( int i = 0; i < sample.getRows().size(); i++ )
		{
			if( i % 2 == 0 )
			{
				out[ 0 ].addRow( sample.getRowAt( i ));
			}
			else 
			{
				out[ 1 ].addRow( sample.getRowAt( i ));
			}
		} 
		
		return out; 
	}
	
	/**
	 * Learn neural Network on training set. 
	 * Learning rules (incl. stopping criterion is set by the learning rule). 
	 * @param trainingSet
	 */
	private void learnNetwork( final DataSet trainingSet )
	{
		getNeuralNetwork().reset(); 
		getNeuralNetwork().learnInNewThread( trainingSet, getLearningRule() );
	}
	
	/**
	 * Report training and validation error. 
	 * @param dataSets
	 * @param out
	 */
	private void reportError( final DataSet[] dataSets, PrintStream out )
	{
		final double trainingError = getError( dataSets[ 0 ] ); 
		final double validationError = getError( dataSets[ 1 ] );
		
		out.println( trainingError + "\t" + validationError ); 
	}
	
	private double getError( final DataSet data )
	{
		getEvaluation().setNeuralNetwork( getNeuralNetwork() );
		return getEvaluation().meanDistance( data ); 
	}
	
	//Getter 
	private NeuralNetwork getNeuralNetwork() { return this.ann; }
	private LearningRule getLearningRule() { return this.learningRule; } 
	private Evaluation getEvaluation() { return this.evaluation; }
}
