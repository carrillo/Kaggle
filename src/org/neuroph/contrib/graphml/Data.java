package org.neuroph.contrib.graphml;

public class Data extends XMLElement {

	public Data( final String key, final String value ) {
		addAttribute( new XMLAttribute( "key", key) );
		addAttribute( new XMLAttribute( "value", value ) );
	}
	
	
	public String toString() {
		
		String out = getStartTag(); 
		out += getValue(); 
		out += getEndTag(); 
		
		return out; 
	}
	
	private String getStartTag() { return new String("<" + getTag() + " " + getAttributes().get( 0 ).toString() + ">"); }
	private String getValue() { return new String(getAttributes().get( 1 ).getValue()); } 
	private String getEndTag() { return new String("</" + getTag() + ">"); }
	
	@Override
	public String getTag() { return new String( "data" ); }
	
	public static void main(String[] args) {
		System.out.println( new Data( "d1", "0.0" ) ); 
	}

}
