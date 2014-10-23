package csvManipulation;

import inputOutput.TextFileAccess;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;

import array.tools.StringArrayTools;

public class ExpandFeatures {
	
	private File fileIn; 
	private File fileOut; 
	
	private ExpandFeatureFunction eff; 
	
	public ExpandFeatures( final File fileIn, final File fileOut ) {
		this.fileIn = fileIn; 
		this.fileOut = fileOut; 
	}
	
	public void setOutputFeatures( final String[] outputFeatures ) {
		eff.setOutputFeatures( getIndexOfFeatures( outputFeatures ) );
	}
	
	public void setFeaturesSkipped( final String[] featuresSkipped ) { 	
		eff.setIndexSkipped( getIndexOfFeatures( featuresSkipped ) );
	}
	
	public HashSet<Integer> getIndexOfFeatures( final String[] foi ) {
		
		final HashSet<String> fs = new HashSet<String>();
		for( String feature : foi ) {
			fs.add( feature ); 
		}
		
		HashSet<Integer> out = new HashSet<Integer>(); 
		try {
			BufferedReader in = TextFileAccess.openFileRead( fileIn ); 
			
			final String[] header = in.readLine().split(",");
			
			for( int i = 0; i < header.length; i++ ) {
				if( fs.contains( header[ i ] ) ) {
					out.add( i ); 
				}
			}
			
		} catch( IOException e ) {
			e.printStackTrace(); 
		}	
		
		return out; 
	}
	
	public void setExpandFeatureFunction( final ExpandFeatureFunction eff ) {
		this.eff = eff; 
	}
	
	public void run() {
		
		try {
			
			BufferedReader in = TextFileAccess.openFileRead( fileIn ); 
			PrintWriter out = TextFileAccess.openFileWrite( fileOut ); 
			
			int line = 0; 
			String[] entries; 
			while( in.ready() ) {
				
				entries = in.readLine().split(","); 
				
				System.out.println( "CurrentLine: " + line ); 
				
				if( line == 0 ) {
					
					
					eff.setOutputSize( entries );
					
					final String[] header = eff.transformHeader( entries  );
					for( int i = 0; i < header.length; i++ ) {
						if( i == 0 ) {
							out.print( header[ i ] );
						} else {
							out.print( "," + header[ i ] );
						} 
					}
					out.println("\n");
					
					//final String lineOut = StringArrayTools.arrayToString( header, ",");
					//System.out.println( lineOut ); 
					//out.println(  lineOut );
					
					
				} else {
					
					//out.println( StringArrayTools.arrayToString( eff.transformData( entries ), "," ) );
					
					final String[] data = eff.transformData( entries  );
					for( int i = 0; i < data.length; i++ ) {
						if( i == 0 ) {
							out.print( data[ i ] );
						} else {
							out.print( "," + data[ i ] );
						} 
					}
					out.println("\n");
				}
				
				
				line++; 
			}
			
			in.close(); 
			out.flush(); 
			out.close(); 
			
			
		} catch (IOException e) {
			e.printStackTrace(); 
		}
		
	}

	public static void main(String[] args) 
	{
		//final File fileIn = new File( "resources/AfSIS/trainingTransformed.csv" );
		//final File fileOut = new File( "resources/AfSIS/trainingTransformedExpanded.csv" );
		
		final File fileIn = new File( "/Users/carrillo/workspace/AlternativeSplicingAnalysis/resources/features/K562_CellPap_noPminZero_noNAs_mean_normalized.csv" );  
		final File fileOut = new File( "/Users/carrillo/workspace/AlternativeSplicingAnalysis/resources/features/K562_CellPap_noPminZero_noNAs_mean_normalized_expanded.csv" );

		final ExpandFeatures ef = new ExpandFeatures( fileIn, fileOut ); 
		
		
		ef.setExpandFeatureFunction( new PolynomialExpansion() );
		//ef.setOutputFeatures( new String[]{ "Ca","P","pH","SOC","Sand","Ptransformed" } );
		//ef.setFeaturesSkipped( new String[]{ "id" } );
		
		ef.setOutputFeatures( new String[]{ "pmin" } );
		ef.setFeaturesSkipped( new String[]{ "sjId", "dominant5SS", "dominant3SS", "minor5SS", "minor3SS", "strand" } );
		
		ef.run(); 
	}

}
