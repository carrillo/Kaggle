package neurophTools;

import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;

/**
 * Evaluates the error of a training data given a distance measure. 
 * 
 * @author carrillo
 *
 */
public class Evaluation 
{
	private NeuralNetwork neuralNetwork; 
	private DistanceMeasure errorMeasure; 
	
	public Evaluation( NeuralNetwork nnet, DistanceMeasure errorMeasure )
	{
		this.neuralNetwork = nnet; 
		this.errorMeasure = errorMeasure; 
	}
	
	public double meanDistance( final DataSet data )
	{
		final double[] distances = distances( data ); 
		return mean( distances ); 
	}
	
	/**
	 * Calculates the distance between predicted and observed output given a specified distance measure. 
	 * @param data
	 * @return
	 */
	private double[] distances( final DataSet data )
	{
		final double[] distance = new double[ data.getRows().size() ];
		
		double[] observed, predicted; 
		for( int m = 0; m < data.getRows().size(); m++ )
		{
			//Get true output 
			observed = data.getRowAt( m ).getDesiredOutput();
			
			//Get predicted output
			getNeuralNetwork().setInput( data.getRowAt( m ).getInput() );
			getNeuralNetwork().calculate();
			predicted = getNeuralNetwork().getOutput();
			
			//System.out.println( "desired: " + Arrays.toString( observed ) + "\t" + "predicted: " + Arrays.toString( predicted ) ); 
			
			//Calculate distance between observed and predicted 
			distance[ m ] = getDistanceMeasure().distance( observed, predicted ); 
		}
		
		return distance; 
	}
	
	/**
	 * Calculates the mean value of a vector. 
	 * @return
	 */
	private double mean( final double[] v )
	{
		double sum = 0; 
		for( int i = 0; i < v.length; i++ )
		{
			sum += v[ i ]; 
		}
		
		return ( sum / v.length );  
	}
	
	//Setter 
	public void setNeuralNetwork( final NeuralNetwork neuralNetwork ) { this.neuralNetwork = neuralNetwork; }
	public void setDistanceMeasure( final DistanceMeasure errorMeasure ) { this.errorMeasure = errorMeasure; }

	//Getter 
	public NeuralNetwork getNeuralNetwork() { return this.neuralNetwork; }
	public DistanceMeasure getDistanceMeasure() { return this.errorMeasure; } 
}
