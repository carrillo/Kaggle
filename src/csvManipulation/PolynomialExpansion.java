package csvManipulation;

import java.util.ArrayList;
import java.util.Arrays;

public class PolynomialExpansion extends ExpandFeatureFunction { 

	@Override
	protected ArrayList<String> getExpandFeatureLabels( final String[] headerRow ) {
		
		final ArrayList<String> relevantFeatures = new ArrayList<String>(); 
		for( int i = 0; i < headerRow.length; i++ ) {
			 
			if( !indexSkipped.contains( i ) && !indexOutputFeatures.contains( i ) ) {
				relevantFeatures.add( headerRow[ i ] ); 
				 
			} 
		}
		
		final ArrayList<String> out = new ArrayList<String>(); 
		for( int i = 0; i < relevantFeatures.size(); i++ ) {
			for( int j = i; j < relevantFeatures.size(); j++ ) { 
				out.add( new String( relevantFeatures.get( i ) + "*" + relevantFeatures.get( j ) ) ); 
			}
		}
		
		return out; 
	}
	
	@Override
	protected ArrayList<String> getExpandFeatureData(String[] dataRow) {
		
		final ArrayList<Double> relevantFeatures = new ArrayList<Double>(); 
		for( int i = 0; i < dataRow.length; i++ ) {
			 
			if( !indexSkipped.contains( i ) && !indexOutputFeatures.contains( i ) ) {
				relevantFeatures.add( Double.parseDouble( dataRow[ i ] ) ); 
				 
			} 
		}
		
		final ArrayList<String> out = new ArrayList<String>(); 
		for( int i = 0; i < relevantFeatures.size(); i++ ) {
			for( int j = i; j < relevantFeatures.size(); j++ ) {
				out.add( String.valueOf( relevantFeatures.get( i ) * relevantFeatures.get( j ) ) ); 
			}
		}
		
		return out; 
	}
	
	public void setOutputSize( final String[] headerRow ) {
		
		final double expandFeatureCount = ( headerRow.length - indexSkipped.size() - indexOutputFeatures.size() ); 
		final int halfsquarePlusDiagonal = (int) Math.round( Math.pow( expandFeatureCount, 2)/2 + expandFeatureCount/2 );  
		
		this.outputSize = ( headerRow.length + halfsquarePlusDiagonal );
	}

}
