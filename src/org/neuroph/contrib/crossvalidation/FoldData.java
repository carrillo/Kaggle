package org.neuroph.contrib.crossvalidation;

import java.util.ArrayList;
import java.util.Collections;

import org.neuroph.core.data.DataSet;
import org.neuroph.util.data.sample.Sampling;

public class FoldData implements Sampling {
	
	private int fold; 
	
	public FoldData( final int fold ) {
		this.fold = fold;  
	}

	@Override
	public DataSet[] sample(DataSet in ) {
		
		//Number of instances within each fold, set the first break 
		final double foldSize = Math.floor( in.size() / (double) fold );
		 
		
		//Create index for shuffling input data. 
		final ArrayList<Integer> indices = getRandomIndexArray( in ); 
		
		//Populate DataSets. 
		final DataSet[] out = getEmptyDataSetArray( in );  
		
		int currentFoldIndex = 0;
		int instanceCount = 0;
		double nextBreak = foldSize;
		
		for( Integer i : indices  ) {
			
			//In case the next bin is reached set index of DataSet[] to next bin. 
			if( instanceCount >= nextBreak && currentFoldIndex < (fold - 1) ) {
				currentFoldIndex++; 
				nextBreak += foldSize; 
			}
			
			//System.out.println( i + "\t" + currentFoldIndex + "\t" + nextBreak ); 
			
			out[ currentFoldIndex ].addRow( in.getRowAt( i ) );
			instanceCount++; 
		}
		
		return out;
	}
	
	/**
	 * Returns an array holding DataSet indices at random order. 
	 * @param in
	 * @return
	 */
	private ArrayList<Integer> getRandomIndexArray( final DataSet in ) {
		final ArrayList<Integer> indices = new ArrayList<Integer>(); 
		for( int i = 0; i < in.size(); i++ ) {
			indices.add( i ); 
		}
		Collections.shuffle( indices );
		return( indices ); 
	}
	
	/**
	 * Return initiated empty DataSet array. 
	 * @param in
	 * @return
	 */
	private DataSet[] getEmptyDataSetArray( final DataSet in ) {
		
		final int inputSize = in.getInputSize(); 
		final int outputSize = in.getOutputSize();  
		
		final DataSet[] out = new DataSet[ fold ];
		
		for( int i = 0; i < fold; i++ ) {
			out[ i ] = new DataSet(inputSize, outputSize); 
		}
		
		return out; 
	}
	
	

}
