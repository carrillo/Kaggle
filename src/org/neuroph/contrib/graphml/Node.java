package org.neuroph.contrib.graphml;
/**
 * XML Element representing a graph node. 
 * 
 * @author fernando carrillo (fernando@carrillo.at)
 *
 */
public class Node extends XMLElement 
{
	public Node( final String id ) {
		addAttribute( new XMLAttribute( "id", id ) );
	}
	
	public String getTag() { return "node"; }
}
