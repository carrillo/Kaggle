package wekaTools;

public class SquareFunction extends MathFunction {

	@Override
	public double run( double x ) 
	{
		return Math.pow( Double.valueOf( x ), 2 );
	}

}
