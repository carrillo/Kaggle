package org.neuroph.contrib.graphml;

/**
 * XML edge element.  
 * 
 * The edge element is characterized by two XML Elements:  
 * 
 * 1. The source node id
 * 2. The target node id 
 * 
 * @author fernando carrillo (fernando@carrillo.at)
 *
 */
public class Edge extends XMLElement 
{
	public Edge( final String sourceId, final String targetId, final String weightKeyId, final String weight ) { 
		addAttribute( new XMLAttribute( "source", sourceId ) );
		addAttribute( new XMLAttribute( "target", targetId ) );
		
		this.appendChild( new Data( weightKeyId, weight ) );
	}
	
	public String getTag() { return "edge"; }
}
