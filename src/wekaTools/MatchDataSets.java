package wekaTools;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import array.tools.StringArrayTools;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.AddValues;
import weka.filters.unsupervised.attribute.Reorder;

/**
 * Matches two data-sets with same features. 
 * @author carrillo
 *
 */
public class MatchDataSets 
{
	protected Instances i1, i2; 
	
	public MatchDataSets( final Instances i1, final Instances i2 )  
	{
		 this.i1 = i1; 
		 this.i2 = i2; 
	}
	
	public void run() throws Exception
	{
		fillMissingFeatures(); 
		orderFeatures(); 
		matchHeader();
	}
	
	/*
	 * Match all features present in i1 and i2. 
	 */
	public void fillMissingFeatures() throws Exception
	{
		final ArrayList<Attribute> missingIn1 = getMissingFeatures( this.i2, this.i1 );
		for( Attribute a : missingIn1 )
		{
			this.i1 = addFeature( this.i1, a ); 
		}
		
		final ArrayList<Attribute> missingIn2 = getMissingFeatures( this.i1, this.i2 );
		for( Attribute a : missingIn2 )
		{
			this.i2 = addFeature( this.i2, a ); 
		}
	}
	
	/*
	 * Adds feature at the end of the Instance 
	 */
	private Instances addFeature( final Instances instance, final Attribute feature ) throws Exception
	{
		Add add = new Add();
		add.setAttributeIndex( "last" );
		add.setAttributeName( feature.name() );
		
		if( !feature.isNumeric() )
		{
			add.setNominalLabels( AttributeManipulations.getLabelString( feature ) );
		}
		
		add.setInputFormat( instance ); 
		return Filter.useFilter(instance, add); 
	}
	
	
	
	/*
	 * Find features missing in query Instances compared to template instances. 
	 */
	public ArrayList<Attribute> getMissingFeatures( final Instances template, final Instances query )
	{
		
		final Attribute[] featuresTempl = InstancesManipulation.getFeatures( template );
		final Attribute[] featuresQuery = InstancesManipulation.getFeatures( query );
		
		HashSet<String> queryHash = new HashSet<String>(); 
		for( Attribute q : featuresQuery )
		{
			queryHash.add( q.name() ); 
		}
		
		ArrayList<Attribute> missingFeatures = new ArrayList<Attribute>(); 
		for( Attribute a : featuresTempl )
		{
			if( !queryHash.contains( a.name() ) )
			{
				missingFeatures.add( a ); 
			}
		}
		
		return missingFeatures; 
	}
	
	/*
	 * Orders features as given by i1. 
	 */
	public void orderFeatures() throws Exception
	{
		Reorder reorder = new Reorder(); 
		reorder.setOptions( getReorderOptions( i1, i2 ) );
		reorder.setInputFormat( i2 ); 
		i2 = Filter.useFilter( i2, reorder );
	}
	
	/*
	 * Generate the string holding the new order of indices. 
	 */
	private String[] getReorderOptions( final Instances template, final Instances toReorder )
	{
		HashMap<String, Integer> featureMap = InstancesManipulation.getFeatureIndexMap( toReorder, 1 ); 
		
		final int[] reorderIndices = new int[ template.numAttributes() ]; 
		for( int n = 0; n < template.numAttributes(); n++ )
		{
			reorderIndices[ n ] = featureMap.get( template.attribute( n ).name() );  
		}
		
		String reorderString = ""; 
		for( int i = 0; i < reorderIndices.length; i++ )
		{
			if( i != 0 )
			{
				reorderString += ","; 
			}
			reorderString += reorderIndices[ i ]; 
		}
		 
		return new String[]{ "-R", reorderString }; 
	}
	
	/*
	 * Matching attributes (header) of both input Instances. 
	 * This is important for nominal values: We have to define the entire set of values 
	 * present in either of the two collections.
	 * 1. Generate the common attributes. 
	 *  
	 */
	public void matchHeader() throws Exception
	{ 
		AddValues addValues = new AddValues();
		addValues.setSort( true );
		for( int n = 0; n < i1.numAttributes(); n++ )
		{		
			if( i1.attribute( n ).isNominal() )
			{	
				addValues.setAttributeIndex( String.valueOf( n + 1 ) );
				 
				final String missingLabelsIn1 = getMissingLabels( i2.attribute( n ), i1.attribute( n ) ); 
				addValues.setLabels( missingLabelsIn1 );
				addValues.setInputFormat( i1 ); 
				i1 = Filter.useFilter( i1, addValues ); 
				 
				
				
				final String missingLabelsIn2 = getMissingLabels( i1.attribute( n ), i2.attribute( n ) ); 
				addValues.setLabels( missingLabelsIn2 );
				addValues.setInputFormat( i2 ); 
				i2 = Filter.useFilter( i2, addValues );
			}
		} 
	}
	
	/*
	 * Get missing labels in query attribute compared to template. 
	 * 1. Check if they have the same name. 
	 * 2. Get set of all labels in query. 
	 * 3. Add template labels not present in the hash set. 
	 * 4. Create comma separated string. 
	 */
	private String getMissingLabels( final Attribute template, final Attribute query )
	{
		
		if( template.name().equals( query.name() ) )
		{
			//Hash all query labels. 
			HashSet<String> queryLabelSet = new HashSet<String>(); 
			for( int i = 0; i < query.numValues(); i++ )
			{
				queryLabelSet.add( query.value( i ) );  
			}
			
			//Get all template values not in query set.
			ArrayList<String> missingLabels = new ArrayList<String>(); 
			for( int i = 0; i < template.numValues(); i++ )
			{
				if( !queryLabelSet.contains( template.value( i ) ) )
				{
					missingLabels.add( template.value( i ) );  
				}
			}
			
			//Generate Label string 
			String out = ""; 
			for( int i = 0; i < missingLabels.size(); i++ )
			{
				if( i != 0 )
				{
					out += ","; 
				}
				out += missingLabels.get( i ); 
			
			}
			
			return out; 
		}
		else 
		{ 
			return null;
		}
	}
	
	public Instances getInstance1() { return this.i1; }
	public Instances getInstance2() { return this.i2; }
	
	public static void main(String[] args) throws Exception 
	{
		final String trainingSet = "resources/titanic/trainClean.csv";
		final String testSet = "resources/titanic/testClean.csv";
		
		Instances i1 = new DataSource( trainingSet ).getDataSet();
		Instances i2 = new DataSource( testSet ).getDataSet();
		
		final String[] nominalClasses = new String[] {
				"Survived"/*,"Pclass"*/,"Surname","Title","Sex","TicketId","CabinDeck","Embarked"
		};
		i1 =  InstancesManipulation.makeNominal( i1, nominalClasses );
		i2 =  InstancesManipulation.makeNominal( i2, nominalClasses );
		
		
		final MatchDataSets mds = new MatchDataSets(i1, i2); 
	}

}
