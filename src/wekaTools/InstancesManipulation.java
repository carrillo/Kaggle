package wekaTools;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

public class InstancesManipulation 
{
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
	 * Make numeric attribute nominal. 
	 */
	public static Instances numeric2nomical( final Instances inData, final int attributeIndex ) throws Exception 
	{ 
		return numeric2nomical(inData, new int[]{ attributeIndex } ); 
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
}
