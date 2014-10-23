package csvManipulation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

public abstract class ExpandFeatureFunction {
	
	protected HashSet<Integer> indexSkipped = new HashSet<Integer>();
	protected HashSet<Integer> indexOutputFeatures = new HashSet<Integer>(); 
	protected int outputSize; 
	
	public String[] transformHeader(String[] headerRow) {		
		final String[] header = new String[ getOutputSize() ];

		int currentIndex = 0; 
		 
		currentIndex = addEntries( getInputFeatures(headerRow), currentIndex, header );
		currentIndex = addEntries( getExpandFeatureLabels( headerRow ), currentIndex, header ); 
		currentIndex = addEntries( getOutputFeatures( headerRow ), currentIndex, header );
		
		return header;
	}
	
	public String[] transformData(String[] dataRow ) {		
		final String[] row = new String[ getOutputSize() ];

		int currentIndex = 0; 
		currentIndex = addEntries( getInputFeatures( dataRow ), currentIndex, row ); 
		currentIndex = addEntries( getExpandFeatureData( dataRow ), currentIndex, row );
		currentIndex = addEntries( getOutputFeatures( dataRow ), currentIndex, row ); 
		
		return row;
	}

	
	private ArrayList<String> getInputFeatures( final String[] row ) {
		final ArrayList<String> out = new ArrayList<String>(); 
		
		for( int i = 0; i < row.length; i++ ) {
			if( !indexOutputFeatures.contains( i ) ) {
				out.add( row[ i ] ); 
			}
		}
		
		return out; 
	}
	
	private ArrayList<String> getOutputFeatures( final String[] row ) {
		
		List<Integer> list = new ArrayList<Integer>( indexOutputFeatures );
		Collections.sort( list );
		
		ArrayList<String> out = new ArrayList<String>(); 
		for( Integer i : list ) { 
			out.add( row[ i ] ); 
		}
		
		return out; 
	}
	
	private int addEntries( final ArrayList<String> entries, Integer startIndex, final String[] header ) {
		for( String s : entries ) { 
			header[ startIndex ] = s; 
			startIndex++; 
		}
		
		return startIndex; 
	}
	
	
	
	protected abstract ArrayList<String> getExpandFeatureLabels( final String[] headerRow );
	protected abstract ArrayList<String> getExpandFeatureData( final String[] dataRow );

	public abstract void setOutputSize( final String[] headerRow ); 
	
	public void setIndexSkipped( HashSet<Integer> indexSkipped ) { this.indexSkipped = indexSkipped; }
	public void setOutputFeatures( HashSet<Integer> outputFeatures ) { this.indexOutputFeatures= outputFeatures; }
	
	public HashSet<Integer> getIndexSkipped() { return this.indexSkipped; }
	public int getOutputSize() { return this.outputSize; } 
	
	
	
}
