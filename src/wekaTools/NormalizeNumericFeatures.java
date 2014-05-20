package wekaTools;
import java.util.ArrayList;
import java.util.HashMap;

import weka.core.DenseInstance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

/**
 * Filter to normalize numerical features between 0 and 1. 
 * Run on training data first to learn the ranges of the parameters 
 * and scale training and test data afterwards. 
 * @author carrillo
 *
 */
public class NormalizeNumericFeatures extends SimpleBatchFilter 
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	HashMap<String, Integer> nameIndexMap; 
	ArrayList<double[]> ranges; 
	
	@Override
	public String globalInfo() {
		return    "A batch filter that scales all numeric features between 0 and 1. "
				+ "Run on training data to determine the range of parameters first. "
				+ "Once the parameters are set filter training and test data.";
		
	}
	
	/*
	 * Set up the input format
	 * 
	 * 1. Get names of all numeric features. 
	 * 2. Determine min and max values of all numeric values
	 *  
	 * 3. Set the output format 
	 * 
	 * (non-Javadoc)
	 * @see weka.filters.SimpleFilter#setInputFormat(weka.core.Instances)
	 */
	public boolean setInputFormat(Instances instanceInfo) throws Exception {
	      super.setInputFormat(instanceInfo);
	 
	      setParameters( instanceInfo );
	      
	      Instances outFormat = new Instances(instanceInfo, 0);
	      setOutputFormat(outFormat);
	 
	      return false;   
	}
	
	/*
	 * Sets 
	 * 1. The names of the numeric features and 
	 * 2. The max and min values used for scaling. 
	 */
	private void setParameters( final Instances trainingData )
	{
		nameIndexMap = new HashMap<String, Integer>(); 
		ranges = new ArrayList<double[]>(); 
		
		int index = 0; 
		for( int i = 0; i < trainingData.numAttributes(); i++ )
		{
			if( trainingData.attribute( i ).isNumeric() )
			{
				nameIndexMap.put( trainingData.attribute( i ).name(), index ); 
				ranges.add( getRange( trainingData.attributeToDoubleArray( i ) ) ); 
				index++; 
			}
		}
	
	}
	
	/*
	 * Returns min and max from values.
	 */
	private double[] getRange( final double[] values ) 
	{
		final double[] minMax = new double[]{ Double.MAX_VALUE, -Double.MAX_VALUE }; 
		
		for( double d : values )
		{
			if( d < minMax[ 0 ] )
			{
				minMax[ 0 ] = d; 
			}
			if( d > minMax[ 1 ] )
				minMax[ 1 ] = d; 
		}
		
		return minMax; 
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {
		return new Instances(inputFormat, 0);
	}

	/*
	 * 
	 * (non-Javadoc)
	 * @see weka.filters.SimpleFilter#process(weka.core.Instances)
	 */
	protected Instances process(Instances instances) throws Exception 
	{
		Instances result = new Instances( determineOutputFormat( instances ), 0 );
		double currentValue;
		double[] currentRange; 
		for( int i = 0; i < instances.numInstances(); i++ )
		{
			double[] values = new double[ result.numAttributes() ]; 
			
			for( int j = 0; j < instances.numAttributes(); j++ )
			{
				currentValue = instances.instance( i ).value( j ); 
				if( nameIndexMap.containsKey( instances.attribute( j ).name() ) ) 
				{
					currentRange = ranges.get( nameIndexMap.get( instances.attribute( j ).name() ) ); 
					values[ j ] = scale( currentValue, currentRange ); 
				}
				else 
				{
					values[ j ] = currentValue; 					
				}
			}
			
			result.add( new DenseInstance( 1, values ) ); 
		}
		
		return result;
	}
	
	/*
	 * Scale between 0 and 1 using (x - min / ( max - min ) ). 
	 * Return NaN if value was not set. 
	 * 
	 */
	private double scale( final double value, final double[] minMax )
	{
		double out; 
		if( Double.isNaN( value ) )
		{
			out = Double.NaN; 
		}
		else 
		{
			out = ( value - minMax[ 0 ] ) / ( minMax[ 1 ] - minMax[ 0 ] );    
		}
		return out; 
	}
}
