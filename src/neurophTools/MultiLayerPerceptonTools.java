package neurophTools;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;

import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.Neuron;
import org.neuroph.nnet.MultiLayerPerceptron;

import sun.security.action.GetLongAction;
import array.tools.DoubleArrayTools;
import array.tools.StringArrayTools;
import weka.core.Summarizable;


public class MultiLayerPerceptonTools 
{
	public static void labelInputOutputNeurons( final MultiLayerPerceptron mlp, 
			final ArrayList<String> inputLabels, final ArrayList<String> outputLabels )
	{
		final Layer inputLayer = mlp.getLayerAt( 0 ); 
		
		labelLayer( inputLayer, inputLabels, true );
		
		final Layer outputLayer = mlp.getLayerAt( mlp.getLayersCount() - 1 );
		
		labelLayer( outputLayer, outputLabels, false );
		
		 
	}
	
	public static void labelLayer( final Layer layer, final ArrayList<String> labels, boolean hasBiasUnit ) 
	{
		int expectedCount = labels.size();
		if( hasBiasUnit )
			expectedCount++; 
		
		if( layer.getNeuronsCount() != expectedCount )
		{
			System.err.println( "Layer and label vector have size mismatch." ); 
		}
		else
		{
			for( int i = 0; i < labels.size(); i++ )
			{
				layer.getNeuronAt( i ).setLabel( labels.get( i ) ); 
			}
			if( hasBiasUnit )
				layer.getNeuronAt( labels.size() ).setLabel( "bias" );
		}
	}
	
	public static String[][] getConnectionWeights( final Layer layerN, final Layer layerNPlus1 )
	{
		 String[] outputNames = getNeuronNames( layerN ); 
		 String[] inputNames = getNeuronNames( layerNPlus1 ); 
		 
		 final String[][] out = new String[ outputNames.length + 1 ][ inputNames.length + 1 ];
		 
		 for( int i = 0; i < inputNames.length; i++ )
			 out[ 0 ][ i + 1 ] = inputNames[ i ]; 
		 
		 
		 Connection[] output; 
		 for( int outputNeuron = 0; outputNeuron < outputNames.length; outputNeuron++ )
		 {
			 out[ outputNeuron + 1 ][ 0 ] = outputNames[ outputNeuron ];  
			 output = layerN.getNeuronAt( outputNeuron ).getOutConnections();
			 for( int inputNeuron = 0; inputNeuron < output.length; inputNeuron++ )
			 {
 
				 String value = output[ inputNeuron ].getWeight().toString(); 
				 out[ outputNeuron + 1 ][ inputNeuron + 1 ] =  value; 
			 }
			 
		 }
		 
		 return out;  
	}
	
	public static String[] getNeuronNames( final Layer l )
	{
		String[] out = new String[ l.getNeuronsCount() ];
		String label; 
		for( int i = 0; i < l.getNeuronsCount(); i++ )
		{
			label = l.getNeuronAt( i ).getLabel(); 
			if( label == null )
			{
				out[ i ] = String.valueOf( i ); 
			}
			else 
			{
				out[ i ] = label; 
			} 
		}
		
		return out; 
	}
}
