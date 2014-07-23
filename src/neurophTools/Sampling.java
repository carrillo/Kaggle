package neurophTools;

import java.util.HashSet;
import java.util.Random;

import org.neuroph.core.data.DataSet;

public class Sampling 
{
	public static DataSet subsampleWithoutReplacement( final DataSet in, final int nrOfSamples )
	{
		DataSet subsample = new DataSet( in.getInputSize(), in.getOutputSize() ); 
		
		final HashSet<Integer> sampledIndices = new HashSet<Integer>();
		Random rnd = new Random(); 
		int currentIndex; 
		while( subsample.getRows().size() < nrOfSamples )
		{
			currentIndex = (int) Math.floor( rnd.nextDouble() * in.getRows().size() );
			if( sampledIndices.contains( currentIndex ) )
				continue; 
			else 
			{
				subsample.addRow( in.getRowAt( currentIndex ) );
				sampledIndices.add( currentIndex ); 
			}
			//System.out.println( currentIndex + "\t" + subsample.getRows().size() ); 
		}
		
		
		return subsample; 
		
	}
}
