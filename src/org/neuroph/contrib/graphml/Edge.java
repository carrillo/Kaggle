package org.neuroph.contrib.graphml;

/**
 * XML element representing an edge from node source to node target. 
 * 
 * @author fernando carrillo (fernando@carrillo.at)
 *
 */
public class Edge extends XMLElement 
{
	public Edge( final String sourceId, final String targetId, final String weight ) { 
		addAttribute( new XMLAttribute( "source", sourceId ) );
		addAttribute( new XMLAttribute( "target", targetId ) );
	}
	
	public String getTag() { return "edge"; }
}
