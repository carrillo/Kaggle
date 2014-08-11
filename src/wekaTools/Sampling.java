package wekaTools;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

import weka.core.Attribute;
import weka.core.Instances;

public class Sampling 
{
	public static Instances subsampleWithoutReplacement( final Instances in, final int nrOfSamples )
	{
		Instances subsample = new Instances( "subsample", getAttributeList( in ), nrOfSamples ); 
		
		final HashSet<Integer> sampledIndices = new HashSet<Integer>();
		Random rnd = new Random(); 
		int currentIndex; 
		while( subsample.numInstances() < nrOfSamples )
		{
			currentIndex = (int) Math.floor( rnd.nextDouble() * in.numInstances() );
			if( sampledIndices.contains( currentIndex ) )
				continue; 
			else 
			{
				subsample.add( in.instance( currentIndex ) ); 
				sampledIndices.add( currentIndex ); 
			}
			System.out.println( currentIndex + "\t" + subsample.numInstances() ); 
		}
		
		
		return subsample; 
		
	}
	
	/**
	 * Returns the Attribute as ArrayList 
	 * @param in
	 * @return
	 */
	private static ArrayList<Attribute> getAttributeList( final Instances in )
	{
		final ArrayList<Attribute> out = new ArrayList<Attribute>(); 
		
		for( int i = 0; i < in.numAttributes(); i++ )
			out.add( in.attribute( i ) ); 
		
		return out; 
	}
}
