package wekaTools;

import java.util.ArrayList;
import java.util.HashSet;

import com.sun.tools.javac.util.List;

import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

public class InstancesManipulation 
{	
	/*
	 * Get attribute index from attribute names
	 */
	public static int getAttributeIndex( final Instances inData, final String name )
	{
		int index = -1; 
		for( int i = 0; i < inData.numAttributes(); i++ ) 
		{
			if( inData.attribute( i ).name().equals( name ) )
			{
				assert ( index == -1 ) : "Attribute name dublicated: " + name;  
				index = i; 
			}
		}
		return index; 
	}
	
	/*
	 * Hash String[] 
	 */
	private static HashSet<String> hashStringArray( final String[] in )
	{
		HashSet<String> out = new HashSet<String>(); 
		for( String s : in ) 
			out.add( s );
		return out; 
		
	}

	/*
	 * Make numeric attribute nominal. Find attributes by name 
	 */
	public static Instances makeNominal( final Instances inData, final String[] attributeNames ) throws Exception
	{
		final HashSet<String> nameHash = hashStringArray( attributeNames );
		
		ArrayList<Integer> indexList = new ArrayList<Integer>(); 
		for( int i = 0; i < inData.numAttributes(); i++  ) 
		{
			if( nameHash.contains( inData.attribute( i ).name() ) && !inData.attribute( i ).isNominal() )
			{ 
				indexList.add( i ); 
			}
		}
		
		return numeric2nomical(inData, indexList); 
	}

	/*
	 * Make numeric attribute nominal. 
	 */
	public static Instances numeric2nomical( final Instances inData, final int attributeIndex ) throws Exception 
	{ 
		return numeric2nomical(inData, new int[]{ attributeIndex } ); 
	}

	/*
	 * Make numeric attribute nominal. 
	 */
	public static Instances numeric2nomical( final Instances inData, final ArrayList<Integer> attributeIndexList ) throws Exception 
	{ 
		final int[] attributeIndexArray = new int[ attributeIndexList.size() ];
		for( int i = 0; i < attributeIndexArray.length; i++ )
			attributeIndexArray[ i ] = attributeIndexList.get( i ); 
		
		return numeric2nomical(inData, attributeIndexArray ); 
	}

	/*
	 * Make numeric attribute nominal. 
	 */
	public static Instances numeric2nomical( final Instances inData, final int[] attributeIndexArray ) throws Exception
	{
		NumericToNominal num2nom = new NumericToNominal(); 		// numeric to nominal filter 
		num2nom.setAttributeIndicesArray( attributeIndexArray );
		num2nom.setInputFormat( inData ); 
		
		Instances outData = Filter.useFilter(inData, num2nom ); 
		return outData; 
	}

	/*
	 * Remove one attribute defined by its index in the data. 
	 */
	public static Instances removeAttribute( final Instances inData, final String attributeName ) throws Exception 
	{ 
		final int attributeIndex = getAttributeIndex(inData, attributeName); 
		return removeAttribute(inData, attributeIndex ); 
	}
	/*
	 * Remove one attribute defined by its index in the data. 
	 */
	public static Instances removeAttribute( final Instances inData, final int attributeIndex ) throws Exception 
	{ 
		return removeAttribute(inData, new int[]{ attributeIndex } ); 
	}
	/*
	 * Remove attributes defined by the index array. 
	 */
	public static Instances removeAttribute( final Instances inData, final int[] attributeIndexArray ) throws Exception
	{
		Remove remove = new Remove(); 							// remove filter
		remove.setAttributeIndicesArray( attributeIndexArray );
		remove.setInputFormat( inData ); 
		Instances outData = Filter.useFilter(inData, remove); 
		return outData; 
	}
	
	/*
	 * Set class attribute using attribute name 
	 */
	public static void setClassAttribute( final Instances inData, final String attributeName ) 
	{
		final int classIndex = getAttributeIndex(inData, attributeName ); 
		inData.setClassIndex( classIndex );
	}

	/*
	 * Get attribute subset of Instances
	 */
	public static Instances subset( final Instances data, final int[] attributeIndex ) throws Exception
	{
		final Remove remove = new Remove();
		remove.setAttributeIndicesArray( attributeIndex );
		remove.setInvertSelection( true );
		remove.setInputFormat( data );
		return Filter.useFilter( data, remove ); 
	}
}
