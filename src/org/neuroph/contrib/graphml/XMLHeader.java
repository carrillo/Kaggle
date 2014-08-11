package org.neuroph.contrib.graphml;

public class XMLHeader 
{
	public static String getHeader() { 
		return getHeader("1.0", "UTF-8"); 
	}
	
	public static String getHeader( final String version, final String encoding ) { 
		String out = "<?xml"; 
		out += " " + new XMLAttribute("version", version );
		out += " " + new XMLAttribute("encoding", encoding ); 
		out += " ?>"; 
		return out;  
	}
}
