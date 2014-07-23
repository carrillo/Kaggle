package wekaTools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import array.tools.ArrayListIntegerTools;
import array.tools.DoubleArrayTools;
import array.tools.StringArrayTools;

import com.sun.tools.javac.util.List;

import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ClassAssigner;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

public class InstancesManipulation 
{	
	/**
	 * Discretizes numeric features. Specify weather the class feature should be discretized as well.  
	 * @throws Exception
	 */
	public static Instances discretizeNumericFeatures( Instances in, final boolean discretizeClassIndex ) throws Exception
	{ 
		//final int classIndex = in.classIndex();
		final String className = in.classAttribute().name(); 
		
		final ClassAssigner classAssigner = new ClassAssigner();
		if( discretizeClassIndex )
		{			
			classAssigner.setClassIndex( "0" );
			classAssigner.setInputFormat( in );
			in = Filter.useFilter(in, classAssigner); 
		}
		
		
		ArrayList<Integer> numericIndices = InstancesManipulation.getNumericFeaturesIndices( in, false );
		for( int i = 0; i < numericIndices.size(); i++ )
		{
			numericIndices.set( i, numericIndices.get( i ) + 1 ); 
		} 
		//System.out.println( Arrays.toString( InstancesManipulation.getFeatureNames( in ) ) ); 
		
		Filter filter = new Discretize( ArrayListIntegerTools.arrayListToString(numericIndices, "," ) );
		filter.setInputFormat( in ); 
		in  = Filter.useFilter( in, filter);
		
		if( discretizeClassIndex )
		{
			setClassAttribute( in, className ); 
			//classAssigner.setClassIndex( String.valueOf( classIndex ) );
			//classAssigner.setInputFormat( in );
			//in = Filter.useFilter( in, classAssigner);
		}
		
		return in; 
	}
	
	/**
	 * Discretizes numeric features. Specify weather the class feature should be discretized as well.  
	 * @throws Exception
	 */
	public static Instances discretizeNumericFeaturesNew( Instances in, final ArrayList<Integer> indices ) throws Exception
	{ 
		final String className = in.classAttribute().name(); 
		final ClassAssigner classAssigner = new ClassAssigner();
		final boolean containsClassIndex = containsClassIndex( in, indices ); 
		if( containsClassIndex )
		{			
			//System.out.println( "Contains class Index" ); 
			classAssigner.setClassIndex( "0" );
			classAssigner.setInputFormat( in );
			in = Filter.useFilter(in, classAssigner); 
		}
		
		
		ArrayList<Integer> indicesForString = new ArrayList<Integer>(); 
		for( int i = 0; i < indices.size(); i++ )
		{
			indicesForString .add( indices.get( i ) + 1 ); 
		} 
		 
		
		Filter filter = new Discretize( ArrayListIntegerTools.arrayListToString( indicesForString, "," ) );
		filter.setInputFormat( in ); 
		in  = Filter.useFilter( in, filter);
		
		if( containsClassIndex )
		{
			setClassAttribute( in, className ); 
		}  
		
		return in; 
	}
	
	/**
	 * Returns true if the class index is contained in the index array
	 * @param in
	 * @param indices
	 * @return
	 */
	private static boolean containsClassIndex( final Instances in, final ArrayList<Integer> indices )
	{
		boolean out =  false; 
		
		for( Integer i : indices )
		{
			if( i == in.classIndex() )
			{
				out = true; 
				break; 
			}
		}
		
		return out; 
	}
	
	/**
	 * Sets all values of given feature with specified value. 
	 * @param in
	 * @param indices
	 * @param value
	 */
	public static void fillFeaturesWithValue( Instances in, final ArrayList<Integer> indices, final double value )
	{
		for( int m = 0; m < in.numInstances(); m++ )
		{
			for( int n = 0; n < indices.size(); n++ )
			{
				in.get( m ).setValue( indices.get( n ), value ); 
			} 
		}
		
		 
	}
	
	/*
	 * Get attribute index from attribute names
	 */
	public static int getFeatureIndex( final Instances inData, final String name )
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
	 * Get all attributes of Instances 
	 */
	public static Attribute[] getFeatures( final Instances inData )
	{
		Attribute[] out = new Attribute[ inData.get( 0 ).numAttributes() ]; 
		for( int i = 0; i < out.length; i++ ) 
		{
			out[ i ] = inData.get( 0 ).attribute( i ); 
			
		}
		return out; 
	}
	
	
	/*
	 * Get summary of all features.
	 */
	public static String[] getFeatureSummary( final Instances inData ) 
	{
		final Attribute[] features = getFeatures( inData );
		final String[] out = new String[ features.length ]; 
		for( int i = 0; i < features.length; i++ )
		{
			out[ i ] = features[ i ].toString(); 
		}
		return out; 
	}
	
	/*
	 * Get names of all features.
	 */
	public static String[] getFeatureNames( final Instances inData ) 
	{
		final Attribute[] features = getFeatures( inData );
		final String[] out = new String[ features.length ]; 
		for( int i = 0; i < features.length; i++ )
		{
			out[ i ] = features[ i ].name(); 
		}
		return out; 
	}
	
	/**
	 * Returns indices of numeric features. 
	 */
	public static ArrayList<Integer> getNumericFeaturesIndices( final Instances in, final boolean excludeClassIndex )
	{
		ArrayList<Integer> indices = new ArrayList<Integer>(); 
		for( int i = 0; i < in.numAttributes(); i++ )
		{
			if( i == in.classIndex() && excludeClassIndex )
				continue; 
			
			if( in.attribute( i ).isNumeric() )
			{
				indices.add( i ); 
			}
		}
		return indices; 
	}
	
	/**
	 * Returns indices of numeric features. 
	 */
	public static ArrayList<Integer> getNominalFeaturesIndices( final Instances in, final boolean excludeClassIndex )
	{
		ArrayList<Integer> indices = new ArrayList<Integer>(); 
		for( int i = 0; i < in.numAttributes(); i++ )
		{
			if( i == in.classIndex() && excludeClassIndex )
				continue; 
			
			
			if( in.attribute( i ).isNominal() )
			{
				indices.add( i ); 
			}
		}
		return indices; 
	}
	
	/**
	 * Extracts data of numeric features as double[]. 
	 * @param in
	 * @param excludeClassVariable
	 * @return
	 */
	public static ArrayList<double[]> getNumericFeaturesAsDoubleArray( final Instances in, final boolean excludeClassVariable )
	{
		final ArrayList<double[]> out = new ArrayList<double[]>(); 
		final ArrayList<Integer> indices = getNumericFeaturesIndices( in, excludeClassVariable );
		
		double[] currentValues; 
		for( int m = 0; m < in.numInstances(); m++ )
		{
			currentValues = new double[ indices.size() ]; 
			for( int n = 0; n < indices.size(); n++ )
			{ 
				currentValues[ n ] = in.get( m ).value( indices.get( n ) ); 
			}
			out.add( currentValues ); 
		}
		
		return out; 
	}
	
	/**
	 * Extracts data of nominal features as double[]. 
	 * @param in
	 * @param excludeClassVariable
	 * @return
	 */
	public static ArrayList<double[]> getNominalFeaturesAsDoubleArray( final Instances in, final boolean excludeClassVariable )
	{
		//final ArrayList<double[]> out = new ArrayList<double[]>(); 
		
		final ArrayList<Integer> indices = getNominalFeaturesIndices( in, excludeClassVariable );
		return getValues( in, indices ); 
		/*
		double[] currentValues; 
		for( int m = 0; m < in.numInstances(); m++ )
		{
			currentValues = new double[ indices.size() ]; 
			for( int n = 0; n < indices.size(); n++ )
			{ 
				currentValues[ n ] = in.get( m ).value( indices.get( n ) ); 
			}
			out.add( currentValues ); 
		}
		
		return out; 
		*/
	}
	
	/**
	 * Get numeric values for class feature. 
	 * @param in
	 * @return
	 */
	public static ArrayList<Double> getNumericClassValue( final Instances in )
	{
		final ArrayList<Double> out = new ArrayList<Double>(); 
		
		final int classIndex = in.classIndex(); 
		for( int m = 0; m < in.numInstances(); m++ )
		{
			out.add( in.get( m ).value( classIndex ) ); 
		}
		
		return out; 
	}
	
	/**
	 * Returns the double value of the features specified by indices. 
	 * If features are not numerical, the internal double representation of the value will be returned. Use getValueStrings if 
	 * nominal values are to be returned. 
	 * @param in
	 * @param indices
	 * @return
	 */
	public static ArrayList<double[]> getValues( final Instances in, final ArrayList<Integer> indices )
	{
		final ArrayList<double[]> out = new ArrayList<double[]>(); 
		double[] currentValues; 
		for( int m = 0; m < in.numInstances(); m++ )
		{
			currentValues = new double[ indices.size() ]; 
			for( int n = 0; n < indices.size(); n++ )
			{ 
				currentValues[ n ] = in.get( m ).value( indices.get( n ) ); 
			}
			out.add( currentValues ); 
		}
		
		return out; 
	}
	
	/**
	 * Returns the string value of the features specified by indices. 
	 *  
	 * @param in
	 * @param indices
	 * @return
	 */
	public static ArrayList<String[]> getStringValues( final Instances in, final ArrayList<Integer> indices )
	{
		final ArrayList<String[]> out = new ArrayList<String[]>(); 
		String[] currentValues; 
		for( int m = 0; m < in.numInstances(); m++ )
		{
			currentValues = new String[ indices.size() ]; 
			for( int n = 0; n < indices.size(); n++ )
			{ 
				currentValues[ n ] = in.get( m ).stringValue( indices.get( n ) ); 
			}
			out.add( currentValues ); 
		}
		
		return out; 
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
	 * Mapping of feature name to index in the Instance file. 
	 */
	public static HashMap<String, Integer> getFeatureIndexMap( final Instances instances, final int startingIndex )
	{
		HashMap<String, Integer> indexMap = new HashMap<String, Integer>(); 
		for( int m = 0; m < instances.numAttributes(); m++ )
		{
			indexMap.put( instances.attribute( m ).name(), ( m + startingIndex ) ); 
		}
		
		return indexMap; 
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
	 * Remove one attribute defined by its feature name. 
	 */
	public static Instances removeAttribute( final Instances inData, final String attributeName ) throws Exception 
	{ 
		final int attributeIndex = getFeatureIndex(inData, attributeName); 
		return removeAttribute(inData, attributeIndex ); 
	}
	/*
	 * Remove one attribute defined by its index in the data. 
	 */
	public static Instances removeAttribute( final Instances inData, final int featureIndex ) throws Exception 
	{ 
		return removeAttribute(inData, new int[]{ featureIndex } ); 
	}
	/*
	 * Remove attributes defined by the index array. 
	 */
	public static Instances removeAttribute( final Instances inData, final int[] featureIndexArray ) throws Exception
	{
		Remove remove = new Remove(); 							// remove filter
		remove.setAttributeIndicesArray( featureIndexArray );
		remove.setInputFormat( inData ); 
		Instances outData = Filter.useFilter(inData, remove); 
		return outData; 
	}
	/*
	 * Remove attributes defined by the index array. 
	 */
	public static Instances removeAttribute( final Instances inData, final ArrayList<Integer> featureIndexArray ) throws Exception
	{
		Remove remove = new Remove(); 							// remove filter
		remove.setAttributeIndicesArray( ArrayListIntegerTools.toArray( featureIndexArray ) );
		remove.setInputFormat( inData ); 
		Instances outData = Filter.useFilter(inData, remove); 
		return outData; 
	}
	
	
	/*
	 * Set class attribute using attribute name 
	 */
	public static void setClassAttribute( final Instances inData, final String featureName ) 
	{
		final int classIndex = getFeatureIndex(inData, featureName ); 
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
