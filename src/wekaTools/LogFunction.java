package wekaTools;

public class LogFunction extends MathFunction {

	@Override
	public double run(double x) 
	{
		return Math.log( x );
	}

	@Override
	public String description() {
		return "log";
	}

}
