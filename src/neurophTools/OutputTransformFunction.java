package neurophTools;

public abstract class OutputTransformFunction {

	public abstract double transform( final double in );
	public abstract double revertTransform( final double in );
	
	/**
	 * Perform transform on double array 
	 * @param in
	 * @return
	 */
	public double[] transform( final double[] in ) {
		final double[] out = new double[ in.length ]; 
		
		for( int i = 0; i < in.length; i++ ) {
			out[ i ] = transform( in[ i ] ); 
		}
		
		return out; 
	}
	
	/**
	 * Perform revert transform on double array 
	 * @param in
	 * @return
	 */
	public double[] revertTransform( final double[] in ) {
		final double[] out = new double[ in.length ]; 
		
		for( int i = 0; i < in.length; i++ ) {
			out[ i ] = revertTransform( in[ i ] ); 
		}
		
		return out; 
	}
}
