package wekaTools;

import java.util.ArrayList;
import java.util.Random;

import org.paukov.combinatorics.Factory;
import org.paukov.combinatorics.Generator;
import org.paukov.combinatorics.ICombinatoricsVector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class ClassifierCharacterization 
{
	/*
	 * Find the smallest error for all combinations of attributes.  
	 */
	public static Instances getBestAttributeCombination( 
			final AbstractClassifier classifier, 
			final Instances data, 
			final boolean verbose 
			) throws Exception
	{
		//Generate an array holding all indices to shuffle (i.e. all except the class index). 
		final int classIndex = data.classIndex(); 
		final Integer[] attributeIndex = new Integer[ data.numAttributes() - 1 ];
		int index = 0; 
		for( int i = 0; i < attributeIndex.length; i++ ) 
		{
			if( index == classIndex )
			{
				index++; 
			} 
			
			attributeIndex[ i ] = index;
			index++; 
		}
		
		// Get best combination for all possible combinations.
		ArrayList<EvaluatedAttributeCombination> evalCombinations = new ArrayList<EvaluatedAttributeCombination>(); 
		for( int i = 0; i < attributeIndex.length; i++ )
		{			
			if( verbose )
				System.out.println( data.numAttributes() + " choose " + (i + 1) + " attributes." ); 
				
			evalCombinations.addAll( combineAttributes( classifier, data, attributeIndex, (i+1) ) );
		}
		
		final EvaluatedAttributeCombination bestCombination = pickBestCombination( evalCombinations );
		
		return( getDataSubset(data, bestCombination.getAttributeIndex() ) ); 
	}
	/*
	 * Evaluate classifier with n choose k combinations of attributes. 
	 */
	public static ArrayList<EvaluatedAttributeCombination> combineAttributes(
			final AbstractClassifier classifier,
			final Instances data, 
			final Integer[] attributeIndex,
			final int chooseNr ) throws Exception
	{
		//Define n
		ICombinatoricsVector<Integer> indexVector = Factory.createVector( attributeIndex );
		//Generate n choose k 
		Generator<Integer> gen = Factory.createSimpleCombinationGenerator( indexVector, chooseNr); 
		
		
		
		// Evaluate all possible combinations. 
		Instances subset;
		Evaluation eval; 
		ArrayList<EvaluatedAttributeCombination> evalList = new ArrayList<EvaluatedAttributeCombination>();  
		for (ICombinatoricsVector<Integer> combination : gen) 
		{
			//Subset data 
			subset = getDataSubset( data, combination );  
			
			//Run evaluation on subset of data 
			eval = evaluate( classifier, subset, 2, new Random() ); 
			
			evalList.add( new EvaluatedAttributeCombination( combination, eval) ); 
		}
		
		return evalList; 
		//ICombinatoricsVector<Integer> bestCombination = pickBestCombination(gen, evalList); 
	}
	
	/*
	 * Subset data with attribute vector. Keep class vector. 
	 */
	private static Instances getDataSubset( final Instances data, final ICombinatoricsVector<Integer> attributeVector ) throws Exception 
	{
		//Initiate subset vector with class attribute index in pos 0. 
		final int attributeCount = attributeVector.getSize(); 
		final int[] dataSubVector = new int[ attributeCount + 1 ];
		dataSubVector[ 0 ] = data.classIndex(); 
		
		//Add remaining indices
		for( int i = 0; i < attributeCount; i++ ) 
			dataSubVector[ i + 1 ] = attributeVector.getValue( i );
		
		final Instances subset = InstancesManipulation.subset( data, dataSubVector ); 
		return subset; 
	}
	
	/*
	 * Evaluate classification 
	 */
	private static Evaluation evaluate( final AbstractClassifier classifier,
			final Instances data, final int kFolds, final Random random ) throws Exception
	{
		final Evaluation eval = new Evaluation( data ); 
		eval.crossValidateModel( classifier, data, kFolds, random );
		return eval; 
	}
	
	/*
	 * Pick the best combination of attributes
	 */
	private static EvaluatedAttributeCombination pickBestCombination( final ArrayList<EvaluatedAttributeCombination> evalCombinations )
	{
		double minError = Double.MAX_VALUE; 
		EvaluatedAttributeCombination bestCombination = null; 
		
		int i = 0; 
		double currentError; 
		for( EvaluatedAttributeCombination combination : evalCombinations ) 
		{
			currentError = combination.getEvaluation().errorRate();  
			if( currentError < minError )
			{
				minError = currentError; 
				bestCombination = combination; 
			}
			i++; 
		}
		
		System.out.println( "Best combination: " + bestCombination );
		return bestCombination; 
	}
}
