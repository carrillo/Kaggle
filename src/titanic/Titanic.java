package titanic;

import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import wekaTools.ClassifierCharacterization;
import wekaTools.InstancesManipulation;

/**
 * Predicting the survival probability of titanic passengers. 
 * The data set is provided by kaggle. 
 * Try different classification methods on the data. 
 * @author carrillo
 *
 */
public class Titanic 
{ 
	private AbstractClassifier classifier; 
	
	public Titanic( final String train ) throws Exception 
	{
		train( train ); 
		
	}
	
	private void train( final String train ) throws Exception
	{ 
		final DataSource source = new DataSource( train );
		
		Instances data = source.getDataSet();
		
		//Make class attribute nominal and define in data 
		final int classIndex = 1;
		data = InstancesManipulation.numeric2nomical( data, classIndex ); 
		data.setClassIndex( classIndex ); 
		
		//Make Pclass attribute nominal
		data = InstancesManipulation.numeric2nomical( data, 2 ); 
		
		
		
		Instances dataSubset = ClassifierCharacterization.getBestAttributeCombination( new NaiveBayes(), data);
		
		
		
		
		//Train naive bayesian classifier. Use updateable classifier for memory efficiency
		final NaiveBayes nb1 = new NaiveBayes(); 
		nb1.buildClassifier( dataSubset );
		setClassifier( nb1 );
		
		/*
		//Use cross validation on a new bayesian classifier
		final NaiveBayes nb2 = new NaiveBayes(); 
		Evaluation eval = new Evaluation( data ); 
		eval.crossValidateModel(nb2, data, 500, new Random() );
		System.out.println( eval.toSummaryString() );
		*/
	}
	
	
	private void setClassifier( final AbstractClassifier classifier ) { this.classifier = classifier; }
	public AbstractClassifier getClassifier() { return this.classifier; }
	
	public static void main(String[] args) throws Exception
	{
		final String trainingSet = "resources/titanic/trainWoName.csv";
		Titanic tt = new Titanic( trainingSet ); 
	}

}
