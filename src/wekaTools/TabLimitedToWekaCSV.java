package wekaTools;

import inputOutput.TextFileAccess;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

import array.tools.StringArrayTools;

public class TabLimitedToWekaCSV 
{
	private File in, out;
	private String separatorIn, missingValueString; 
	
	public TabLimitedToWekaCSV( final File in, final File out, 
			final String separatorIn, final String missingValueString )
	{
		this.in = in; 
		this.out = out; 
		
		this.separatorIn = separatorIn; 
		this.missingValueString = missingValueString; 
	}
	
	/**
	 * Replace field separator by 
	 * @throws IOException
	 */
	public void run() throws IOException 
	{
		BufferedReader in = TextFileAccess.openFileRead( this.in );
		PrintWriter out = TextFileAccess.openFileWrite( this.out ); 
		
		String[] entries; 
		while( in.ready() )
		{
			entries = in.readLine().replace( this.missingValueString, "?" ).split( this.separatorIn );
			out.println( StringArrayTools.arrayToString(entries, ",") ); 
		}
		
		in.close();
		out.flush(); 
		out.close();
	}
	
	
	
	public static void main(String[] args) throws IOException
	{
		final File in = new File( "/Volumes/passport/alternativeSplicingData/k562/features/joinedFeatures/K562_CellPap_noPminZero_noNAs_mean.txt" );
		final String sep = new String( "\t" ); 
		final File out = new File( "/Volumes/passport/alternativeSplicingData/k562/features/joinedFeatures/K562_CellPap_noPminZero_noNAs_mean.csv" );
		final String missingValueString = new String( "NA" ); 
		
		 
		
		TabLimitedToWekaCSV tabToCsv = new TabLimitedToWekaCSV( in, out, sep, missingValueString );
		tabToCsv.run(); 
	}

}
