package org.neuroph.contrib.graphml;

public class Key extends XMLElement 
{
	public Key( final String idValue, final String forValue, 
			final String attrNameValue, final String attrTypeValue )
	{
		addAttribute( new XMLAttribute( "id", idValue ) );
		addAttribute( new XMLAttribute( "for", forValue ) );
		addAttribute( new XMLAttribute( "attr.name", attrNameValue ) );
		addAttribute( new XMLAttribute( "attr.type", attrTypeValue ) );
	}
	
	public String getTag() { return "key"; }
	
	public static void main(String[] args) {
		System.out.println( new Key( "d0", "edge", "weight", "double" ) ); 
	}
}