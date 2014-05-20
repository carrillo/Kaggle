package libsvmTools;

import java.util.ArrayList;
import java.util.HashMap;

import array.tools.ArrayListStringTools;
import array.tools.StringArrayTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.SMO;

/**
 * Makes modification of libsvm parameters easier. 
 * 
 * @author carrillo
 *
 */
public class ClassifierParameters 
{
	private ArrayList<String> parametes = new ArrayList<String>(); 
	private ArrayList<String> values = new ArrayList<String>(); 
	private HashMap<String, Integer> indexMap = new HashMap<String, Integer>(); 
	
	/*
	 * Initiate with default parameters. 
	 */
	public ClassifierParameters( AbstractClassifier classifier ) 
	{ 
		Integer currentIndex = 0; 
		for( int i = 0; i < classifier.getOptions().length; i++ )
		{
			if( i % 2 == 0 )
			{
				parametes.add( classifier.getOptions()[ i ] );  
				indexMap.put( parametes.get( currentIndex ), currentIndex ); 
				currentIndex++; 
			}
			else 
			{
				values.add( classifier.getOptions()[ i ] );  
			}
		}
	}
	
	/*
	 * Returns the current value of the specified paramter. 
	 */
	public String getValue( final String paramterId )
	{
		return this.values.get( indexMap.get( paramterId ) ); 
	}
	
	/*
	 * Sets the given parameter to the given value. 
	 */
	public void setValue( final String parameterId, final String newValue )
	{
		final int index = this.indexMap.get( parameterId ); 
		this.values.set( index, newValue );  
	}
	/*
	 * Sets the given parameter to the given value. 
	 */
	public void setValue( final String parameterId, final int newValue )
	{
		final int index = this.indexMap.get( parameterId ); 
		this.values.set( index, String.valueOf( newValue ) );  
	}
	/*
	 * Sets the given parameter to the given value. 
	 */
	public void setValue( final String parameterId, final double newValue )
	{
		final int index = this.indexMap.get( parameterId ); 
		this.values.set( index, String.valueOf( newValue ) );  
	}
	
	
	/*
	 * Returns the String array in the form parameter,value,parameter,value.... used in libSVM. 
	 */
	public String[] getLibSVMOptionStringArray()
	{
		final String[] out = new String[ ( getParamters().size() * 2 ) ];
		int currentIndex = 0; 
		for( int i = 0; i < getParamters().size(); i++ )
		{
			out[ currentIndex ] = parametes.get( i ); 
			currentIndex++; 
			out[ currentIndex ] = values.get( i ); 
			currentIndex++; 
		}
		return out; 
	}
	
	//Getter 
	public ArrayList<String> getParamters() { return this.parametes; } 
	public ArrayList<String> getValues() { return this.values; }
	
	public static void main(String[] args) 
	{
		ClassifierParameters param = new ClassifierParameters( new SMO() ); 
		
		System.out.println( ArrayListStringTools.arrayListToString( param.parametes) ); 
		System.out.println( ArrayListStringTools.arrayListToString( param.values) );
		//param.setValue("-K", 0 );
		System.out.println( ArrayListStringTools.arrayListToString( param.values) );
		
		System.out.println( StringArrayTools.arrayToString( param.getLibSVMOptionStringArray() ) );
		
		System.out.println( param.getValue( "-K" ) ); 
	}
}

