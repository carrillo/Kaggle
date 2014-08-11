package neurophTools;

public class MeanSquaredError extends DistanceMeasure {

	/**
	 * Calcluate the mean squared error. 
	 * 
	 * for vectors v1 and v2 of length I calculate 
	 * 
	 * MSE = 1/I * Sum[ ( v1_i - v2_i )^2 ]
	 */
	@Override
	public double distance(double[] vector1, double[] vector2) {
		if( vector1.length == vector2.length )
		{
			double sum = 0; 
			for( int i = 0; i < vector1.length; i++ )
			{
				sum += Math.pow( vector1[ i ] - vector2[ i ] , 2 );
			}
			
			return ( sum / vector1.length ); 
		}
		else 
		{
			throw new IllegalArgumentException(); 
		}
	}
	
	
	
	public static void main(String[] args) 
	{
		final double[] v1 = new double[]{ 0.02, 0.1 }; 
		final double[] v2 = new double[]{ 0.01, 0.1 }; 

		
		MeanSquaredError error = new MeanSquaredError();
		System.out.println( error.distance( v1, v2 ) );

	}


}
