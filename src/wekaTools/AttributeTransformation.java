package wekaTools;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;

import array.tools.ArrayListIntegerTools;
import array.tools.DoubleArrayTools;
import weka.core.Attribute;
import weka.core.Instances;

public class AttributeTransformation 
{
	
	/**
	 * Add second order polynomial combination of numeric features. 
	 * For 3 numeric features x1,x2,x3 this results in 
	 * x1*x1, x1*x2, x1*x3
	 * x2*x2, x2*x3
	 * x3*x3
	 * 
	 * 1. Loop through all combinations of numeric features. The order does not matter, therefore consider half the matrix including the diagonal. 
	 * 2. Generate new feature with combined name. Insert at the end of the features.
	 * 3. Fill with values.  
	 */
	public static Instances addPolynomialCombinations( Instances instances ) throws Exception
	{
		final ArrayList<Integer> numericIndices = InstancesManipulation.getNumericFeaturesIndices( instances, true );
		
		String featureName = "";
		Double valueA, valueB; 
		for( int i = 0; i < numericIndices.size(); i++ )
		{
			for( int j = i; j < numericIndices.size(); j++ )
			{				
				 featureName = instances.attribute( numericIndices.get( i ) ).name() + "*" + instances.attribute( numericIndices.get( j ) ).name();
				 instances.insertAttributeAt( new Attribute( featureName ), instances.numAttributes() );
				 
				 System.out.println( featureName ); 
				 
				 for( int m = 0; m < instances.numInstances(); m++ )
				 {
					 valueA = instances.get( m ).value( numericIndices.get( i ) ); 
					 valueB = instances.get( m ).value( numericIndices.get( j ) );
					 
					 //System.out.println( featureName + "\t" + valueA + "\t" + valueB ); 
					 
					 instances.get( m ).setValue( instances.numAttributes() - 1, ( valueA * valueB ) );
					 //System.out.println( instances.get( m ) ); 
				 }
			}
			 
		}
		
		//return InstancesManipulation.removeAttribute(instances, ArrayListIntegerTools.toArray(numericIndices) );
		return instances;
	}
	
	/**
	 * Add second order polynomial combination of numeric features. 
	 * For 3 numeric features x1,x2,x3 this results in 
	 * x1*x1, x1*x2, x1*x3
	 * x2*x2, x2*x3
	 * x3*x3
	 * 
	 * 1. Loop through all combinations of numeric features. The order does not matter, therefore consider half the matrix including the diagonal. 
	 * 2. Generate new feature with combined name. Insert at the end of the features.
	 * 3. Fill with values.  
	 */
	public static Instances addSimpleDivisionCombinations( Instances instances ) throws Exception
	{
		final ArrayList<Integer> numericIndices = InstancesManipulation.getNumericFeaturesIndices( instances, true );
		
		String featureName = "";
		Double valueA, valueB; 
		for( int i = 0; i < numericIndices.size(); i++ )
		{
			for( int j = i + 1; j < numericIndices.size(); j++ )
			{				
				 featureName = instances.attribute( numericIndices.get( i ) ).name() + ":" + instances.attribute( numericIndices.get( j ) ).name();
				 instances.insertAttributeAt( new Attribute( featureName ), instances.numAttributes() );
				 
				 System.out.println( featureName ); 
				 
				 for( int m = 0; m < instances.numInstances(); m++ )
				 {
					 valueA = instances.get( m ).value( numericIndices.get( i ) ); 
					 valueB = instances.get( m ).value( numericIndices.get( j ) );
					 
					 //System.out.println( featureName + "\t" + valueA + "\t" + valueB ); 
					 
					 instances.get( m ).setValue( instances.numAttributes() - 1, ( valueA / valueB ) );
					 //System.out.println( instances.get( m ) );
					 
 					 if( Double.isNaN( instances.get( m ).value( instances.numAttributes() - 1 )  ) )
					 {
						 System.err.println( "NA produced." ); 
						 System.err.println( featureName + "\t" + valueA + "\t" + valueB ); 
					 }
				 }
			}
			 
		}
		
		//return InstancesManipulation.removeAttribute(instances, ArrayListIntegerTools.toArray(numericIndices) );
		return instances;
	}
	
	/**
	 * Adds a transformed version of the given feature to the instances. 
	 * Transformation is performed by the function extending MathFunction 
	 */
	public static void addTransformedFeature( final Instances inData, final String featureToTransform, final MathFunction function )
	{
		//Add new feature to data 
		final String newFeatureName = new String( featureToTransform + "_" + function.description() );
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
	

	/**
	 * Expands a nominal feature with n classes into n numerical features with 0 and 1. 
	 * 
	 * 1. Extract values from nominal class. 
	 * 2. Add a new feature for each value to the end of the data
	 * 3. Hash value index pairs. 
	 * Feature names are expressed   
	 * @param in
	 * @param index
	 * @return
	 */
	public static Instances expandNominalFeature( Instances in, final int index ) throws Exception
	{
		final Attribute feature = in.attribute( index );
		Enumeration<String> values = feature.enumerateValues();
		
		//Add features and keep value index pairs. 
		String newFeatureName;
		int newIndex; 
		HashMap<String, Integer> valueIndexHash = new HashMap<String, Integer>(); 
		while( values.hasMoreElements() )
		{
			String value = values.nextElement(); 
			newFeatureName = feature.name() + "_" + value;
			newIndex = in.numAttributes(); 
			
			valueIndexHash.put( value, newIndex );
			
			in.insertAttributeAt( new Attribute( new String( newFeatureName ) ), newIndex ); 
		}
		
		InstancesManipulation.fillFeaturesWithValue(in, new ArrayList<Integer>( valueIndexHash.values() ), 0 );
		
		String currentValue; 
		for( int m = 0; m < in.numInstances(); m++ )
		{
			currentValue = in.get( m ).stringValue( index );
			 
			in.get( m ).setValue( valueIndexHash.get( currentValue ), 1 ); 
		}
		 
		return InstancesManipulation.removeAttribute( in, index ); 
	}
	
	/**
	 * Expand all nominal features such that a nominal feature with n values is represented by n features with binary value. 
	 * @param instances
	 * @param excludeClassIndex
	 * @return
	 * @throws Exception
	 */
	public static Instances expandNominalFeatures( final Instances instances, final boolean excludeClassIndex, final boolean verbose ) throws Exception
	{
		long time = System.currentTimeMillis(); 
		if( verbose )
		{
			System.out.println( "Expanding nominal features." );  
		}
		
		ArrayList<Integer> indices = InstancesManipulation.getNominalFeaturesIndices( instances, excludeClassIndex ); 
		
		Instances out = null; 
		for( int index : indices )
		{
			if( verbose )
			{
				System.out.println( "Current feature: " + instances.attribute( index ).name() ); 
			}
			out = expandNominalFeature( instances, index ); 
		}
		
		if( verbose )
		{
			System.out.println( "Expanding nominal features. Done in " + ( System.currentTimeMillis() - time )/100 + "s." ); 
		}
		
		return out; 
	}
	
	
}
