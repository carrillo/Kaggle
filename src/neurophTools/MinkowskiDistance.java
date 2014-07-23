package neurophTools;

/**
 * Generalized distance of order p between two vectors of the same dimension N. 
 * 
 * x=(x_1,x_2,....,x_n), y=(y_1,y_2,...,y_n)
 * 
 *  [ SUM_1^N( | x_i - y_i|^p ) ]^(1/p). 
 *  
 *  p = 1: Manhatten distance 
 *  p = 2: Euclidian distance
 *  
 *  I implemented the algorithm in the non-vectorized form to limit the number of depencies. 
 * @author carrillo
 *
 */
public class MinkowskiDistance extends DistanceMeasure 
{
	private double order; 
	
	public MinkowskiDistance( final double order )
	{
		this.order = order; 
	}
	
	@Override
	public double distance( final double[] vector1, final double[] vector2 ) 
	{ 
		if( vector1.length == vector2.length )
		{
			double sum = 0; 
			for( int i = 0; i < vector1.length; i++ )
			{
				sum += Math.pow( Math.abs( vector1[ i ] - vector2[ i ] ), order );
			}
			
			return Math.pow( sum, ( 1 / order ) );
		}
		else 
		{
			throw new IllegalArgumentException(); 
		}
	}

	public static void main(String[] args) 
	{
		final double[] v1 = new double[]{ 0, 1, 1 }; 
		final double[] v2 = new double[]{ 1, 0, 0 }; 
		
		final double order = 2;
		
		MinkowskiDistance minkowskiDistance = new MinkowskiDistance( order );
		System.out.println( minkowskiDistance.distance( v1, v2 ) ); 
	}
}
