package wekaTools;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import array.tools.DoubleArrayTools;

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
	 * Discretizes the class feature. 
	 * @throws Exception
	 */
	public static Instances discretizeClassFeature( Instances in ) throws Exception
	{
		final int classIndex = in.classIndex(); 
		final ClassAssigner classAssigner = new ClassAssigner();
		classAssigner.setClassIndex( "0" );
		classAssigner.setInputFormat( in );
		in = Filter.useFilter( in, classAssigner); 
		
		Filter filter = new Discretize( String.valueOf( classIndex ) );
		filter.setInputFormat( in ); 
		in = Filter.useFilter( in, filter);
		
		classAssigner.setClassIndex( String.valueOf( classIndex ) );
		classAssigner.setInputFormat( in );
		return Filter.useFilter( in, classAssigner);
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
	
	/*
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
	
	/**
	 * Adds a squared version of the given feature to the instances. 
	 */
	public static void addTransformedFeature( final Instances inData, final String featureToTransform, final MathFunction function )
	{
		//Add new feature to data 
		final String newFeatureName = new String( featureToTransform + "Squared" ); 
		inData.insertAttributeAt( new Attribute( new String( newFeatureName ) ), inData.numAttributes() );
		
		final int vIndex = InstancesManipulation.getFeatureIndex( inData, featureToTransform );
		final int newIndex = InstancesManipulation.getFeatureIndex( inData, newFeatureName );
		
		double currentValue; 
		for( int i = 0; i < inData.numInstances(); i++ )
		{
			currentValue = inData.instance( i ).value(  vIndex );
			inData.instance( i ).setValue( newIndex, function.run( currentValue ) );
		}
		
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
