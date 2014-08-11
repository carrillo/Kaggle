package org.neuroph.contrib.graphml;


import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import org.neuroph.contrib.examples.XORExample;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;

public class GraphmlExport 
{
	private NeuralNetwork ann;
	private XMLElement graphml; 
	
	public GraphmlExport( final NeuralNetwork ann ) {
		this.ann = ann; 
		labelUnmarkedNeurons( this.ann );  
	}
	
	/**
	 * Parses neural network to graphml object. 
	 */
	public void parse() {
		graphml = new Graphml(); 
		graphml.appendChild( createGraph( this.ann ) );
	}
	
	
	/**
	 * Print graphml representation to PrintStream
	 * @param out
	 */
	public void print( final PrintStream out ) { 
		out.println( XMLHeader.getHeader() );
		out.println( graphml ); 
	}
	
	/**
	 * Create XML graph from neuroph neural network. 
	 * @param ann
	 * @return
	 */
	private Graph createGraph( final NeuralNetwork ann ) {
		String id = ann.getLabel();
		if( id == null || id.length() == 0 ) { 
			id = "defaultId"; 
		}
		Graph graph = new Graph( id );
		graph.addNetwork( ann ); 
		
		return graph; 
	}
	
	/**
	 * Labels neurons which are yet unlabelled. 
	 *  
	 * @param ann
	 */
	private void labelUnmarkedNeurons( final NeuralNetwork ann ) {
	
		for( int layer = 0; layer < ann.getLayersCount(); layer++ ) {
			
			int neuronCount = 0; 
			for( Neuron neuron : ann.getLayerAt( layer ).getNeurons()  ) {
			
				labelNeuron(layer, neuronCount, neuron);
				neuronCount++;  
			}
		}
	}
	
	/**
	 * Labels unlabelled neuron according to following rules. 
	 * 1. If Input neuron: "Input-[neuronCount]"
	 * 2. If Bias neuron: "L[layer]-bias"
	 * 3. otherwise: L[layer]-[neuronCount]  
	 * @param layer
	 * @param neuronCount
	 * @param neuron
	 */
	private void labelNeuron( final int layer, final int neuronCount, final Neuron neuron ) { 
		if( neuron.getLabel() == null ) {
			
			if( neuron.getClass() == InputNeuron.class ) {
				neuron.setLabel( "Input-" + neuronCount );
			} else if ( neuron.getClass() == BiasNeuron.class ) {
				neuron.setLabel( "L" + layer + "-bias"  );
			} else { 
				neuron.setLabel( "L" + layer + "-" + neuronCount  );
			}
			
		}
	}
	
	//Getter
	public NeuralNetwork getNeuralNetwork() { return this.ann; }
	public XMLElement getGraphml() { return this.graphml; } 
	
	public static void main(String[] args) throws IOException
	{
		GraphmlExport ge = new GraphmlExport( XORExample.getNetwork( false ) );
		ge.parse(); 
		
		File file = new File("/Users/carrillo/Desktop/temp/test.graphml"); 
		file.createNewFile(); 
		
		ge.print( new PrintStream( file ) );
	}

}
