package wekaTools;

import java.util.ArrayList;

import org.apache.commons.lang3.ArrayUtils;

import weka.core.Attribute;
import weka.core.Instances;

public class WekaToNeurophTools {

	/**
	 * Returns the numeric features as a double array per instance. 
	 * @param instances
	 * @return
	 */
	public static ArrayList<double[]> getNumericInput( final Instances instances )
	{
		final ArrayList<double[]> out = new ArrayList<double[]>(); 
		final ArrayList<Integer> indices = InstancesManipulation.getNumericFeaturesIndices( instances, true );
		
		double[] currentValues; 
		for( int m = 0; m < instances.numInstances(); m++ )
		{
			currentValues = new double[ indices.size() ]; 
			for( int n = 0; n < indices.size(); n++ )
			{ 
				currentValues[ n ] = instances.get( m ).value( indices.get( n ) ); 
			}
			out.add( currentValues ); 
		}
		
		return out; 
	}
	
	/**
	 * Returns the labels of the numeric input features. 
	 * @param instances
	 * @return
	 */
	public static ArrayList<String> getNumericInputLabels( final Instances instances )
	{
		final ArrayList<String> out = new ArrayList<String>(); 
		final ArrayList<Integer> indices = InstancesManipulation.getNumericFeaturesIndices( instances, true );
		
		for( Integer index : indices )
		{
			out.add( instances.attribute( index ).name() ); 
		}
		
		return out;
	}
	
	public static ArrayList<double[]> getNominalOutput( final Instances instances ) throws Exception
	{
		 
		final ArrayList<Integer> nominalIndices = InstancesManipulation.getNominalFeaturesIndices( instances, true ); 
		
		int startingIndex = instances.numAttributes() - nominalIndices.size(); 
		final Instances expandedInstances = AttributeTransformation.expandNominalFeatures( instances, true, true );  
		
		ArrayList<Integer> indicesToKeep = new ArrayList<Integer>(); 
		while( startingIndex < expandedInstances.numAttributes() )
		{
			indicesToKeep.add( startingIndex ); 
			startingIndex++; 
		}
		 
		
		return InstancesManipulation.getValues( expandedInstances, indicesToKeep ); 
	}
	
	public static ArrayList<double[]> getNumericAndNominalOuput( final Instances instances ) throws Exception
	{
		final ArrayList<double[]> numericValues = getNumericInput( instances ); 
		final ArrayList<double[]> nominalValues = getNominalOutput( instances );  
		
		final ArrayList<double[]> out = new ArrayList<double[]>(); 
		for( int i = 0; i < numericValues.size(); i++ )
		{
			out.add( ArrayUtils.addAll( numericValues.get( i ), nominalValues.get( i ) ) ); 
		}
		
		return out; 
	}
	
	/**
	 * Returns the output values to train the neural network. 
	 * 
	 * If the class variable is numeric the single double value is returned. 
	 * @param instances
	 * @return
	 */
	public static ArrayList<double[]> getOuput( final Instances instances ) throws Exception
	{
		final Attribute classFeature = instances.classAttribute();
		final int classIndex = instances.classIndex(); 
		//System.out.println( "Class index " + classIndex + " name: " + instances.classAttribute().name() ); 
		
		ArrayList<double[]> out = new ArrayList<double[]>(); 
		if( classFeature.isNumeric() )
		{
			for( int m = 0; m < instances.numInstances(); m++ )
			{
				 out.add( new double[]{ instances.get( m ).value( classIndex ) } ); 
			}
		}
		else if( classFeature.isNominal() )
		{
			int attributeCountBefore = instances.numAttributes() - 1; 
			final Instances expandedInstances = AttributeTransformation.expandNominalFeature( instances, classIndex );
			
			final ArrayList<Integer> indicesToKeep = new ArrayList<Integer>(); 
			while( attributeCountBefore < expandedInstances.numAttributes() )
			{
				indicesToKeep.add( attributeCountBefore ); 
				attributeCountBefore++; 
			}
			
			out = InstancesManipulation.getValues( expandedInstances, indicesToKeep); 
		}
		
		return out; 
	}
	
	/**
	 * Return labels of output variable. 
	 * @param instances
	 * @return
	 */
	public static ArrayList<String> getOutputLabels( final Instances instances ) throws Exception
	{
		final Attribute classFeature = instances.classAttribute();
		final int classIndex = instances.classIndex();  
		
		ArrayList<String> out = new ArrayList<String>(); 
		if( classFeature.isNumeric() )
		{
			out.add( classFeature.name() ); 
		}
		else if( classFeature.isNominal() )
		{
			int attributeCountBefore = instances.numAttributes() - 1; 
			final Instances expandedInstances = AttributeTransformation.expandNominalFeature( instances, classIndex );
			 
			while( attributeCountBefore < expandedInstances.numAttributes() )
			{
				out.add( expandedInstances.attribute( attributeCountBefore ).name() ); 
				
				attributeCountBefore++; 
			}
			
		}
		
		return out;
	}
	
	

}
