package titanic;

import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.Loader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import wekaTools.InstancesManipulation;

/**
 * Predicting the survival probability of titanic passengers. 
 * The data set is provided by kaggle. 
 * Try different classification methods on the data. 
 * @author carrillo
 *
 */
public class TitanicTrain 
{ 
	
	public TitanicTrain( final String train ) throws Exception 
	{
		parsePassengerList( train ); 
	}
	
	private void parsePassengerList( final String train ) throws Exception
	{ 
		final DataSource source = new DataSource( train );
		
		Instances data = source.getDataSet();
		
		//Make class attribute nominal and define in data 
		final int classIndex = 1;
		data = InstancesManipulation.numeric2nomical( data, classIndex ); 
		data.setClassIndex( classIndex );
		
		//Make Pclass attribute nominal
		data = InstancesManipulation.numeric2nomical( data, 2 ); 
		
		data = InstancesManipulation.removeAttribute( data, 10 );
		data = InstancesManipulation.removeAttribute( data, 9 );
		
		//data = InstancesManipulation.removeAttribute( data, 8 );
		//data = InstancesManipulation.removeAttribute( data, 7 );
		data = InstancesManipulation.removeAttribute( data, 6 );
		//data = InstancesManipulation.removeAttribute( data, 5 );
		//data = InstancesManipulation.removeAttribute( data, 4 );
		//data = InstancesManipulation.removeAttribute( data, 3 );
		//data = InstancesManipulation.removeAttribute( data, 2 );
		
		
		data = InstancesManipulation.removeAttribute( data, 0 );
		
		
		//Train naive bayesian classifier. Use updateable classifier for memory efficiency
		final NaiveBayes nb1 = new NaiveBayes(); 
		nb1.buildClassifier( data ); 
		
		//Use cross validation on a new bayesian classifier
		final NaiveBayes nb2 = new NaiveBayes(); 
		Evaluation eval = new Evaluation( data ); 
		eval.crossValidateModel(nb2, data, 500, new Random() );
		System.out.println( eval.toSummaryString() );
	}
	
	
	public static void main(String[] args) throws Exception
	{
		final String trainingSet = "resources/titanic/trainWoName.csv";
		TitanicTrain tt = new TitanicTrain( trainingSet ); 
		

	}

}
