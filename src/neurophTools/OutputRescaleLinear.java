package neurophTools;

import java.util.Arrays;

public class OutputRescaleLinear extends OutputTransformFunction {

	private double min; 
	private double max;
	private double range; 
	
	private double observedMin; 
	private double observedMax; 
	private double observedRange; 
	
	public OutputRescaleLinear( final double[] output, final double min, final double max ) {
		this.min = min; 
		this.max = max; 
		this.range = ( max - min ); 
		
		learnObservedExtrema( output );
	}
	
	/**
	 * Find min and max of the data. 
	 * @param output
	 */
	private void learnObservedExtrema( final double[] output ) {
		
		observedMin = Double.MAX_VALUE;
		observedMax = -Double.MAX_VALUE;
		
		for( double d : output ) {
			if( d > observedMax ) {
				observedMax = d; 
			}
			if( d < observedMin ) {
				observedMin = d; 
			}
		} 
		
		observedRange = observedMax - observedMin; 
	}
	
	
	/**
	 * Calculate the transform 
	 * range0 = observedMax - observedMin 
	 * range1 = max - min 
	 * 
	 *     ( x - observedMin ) * range
	 * y = ---------------------------- + min 
	 *          observedRange  
	 * 
	 * 
	 */
	@Override
	public double transform( double in ) {
		return ( ( ( in - observedMin ) * range ) / observedRange ) + min;
	}

	@Override
	/**
	 * Calculate the inverse of the transform function
	 * range0 = observedMax - observedMin
	 * range1 = max - min 
	 * 
	 *     ( x - min ) * observedRange
	 * y = ---------------------------- + observedMin 
	 *               range  
	 */
	public double revertTransform(double in) {
		return ( ( ( in - min ) * observedRange ) / range ) + observedMin;
	}
	
	public static void main(String[] args) {
		final double[] output = new double[]{ 1, -1, 5, 0.2, -6, -0.03, 2, 3, 0 }; 
		OutputRescaleLinear rescale = new OutputRescaleLinear(output, -1, 1); 
		
		final double[] transform = rescale.transform( output );
		final double[] revertTransform = rescale.revertTransform( transform );
		
		System.out.println( Arrays.toString( output ) ); 
		System.out.println( Arrays.toString( transform ) );
		System.out.println( Arrays.toString( revertTransform ) );
	}

}
