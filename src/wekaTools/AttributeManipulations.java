package wekaTools;

import weka.core.Attribute;

public class AttributeManipulations {

	/**
	 * Get labels from nominal attibute. 
	 * @param attribute
	 * @return
	 */
	public static String[] getLabels( final Attribute attribute )
	{
		final int numValues = attribute.numValues(); 
		if( numValues == 0 )
		{
			return null; 
		}
		else
		{
			String[] out = new String[ attribute.numValues() ];
			for( int i = 0; i < attribute.numValues(); i++ )
			{
				out[ i ] = attribute.value( i ); 
			}
			return out; 
		}
	}
	
	/**
	 * Get labels from nominal attibute. 
	 * @param attribute
	 * @return
	 */
	public static String getLabelString( final Attribute attribute )
	{
		final int numValues = attribute.numValues(); 
		if( numValues == 0 )
		{
			return null; 
		}
		else
		{
			String out = "";
			for( int i = 0; i < attribute.numValues(); i++ )
			{
				if( i != 0 )
					out += ","; 
				out += attribute.value( i ); 
			}
			return out; 
		}
	}

}
