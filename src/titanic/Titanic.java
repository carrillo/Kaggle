package titanic;

import inputOutput.TextFileAccess;

import java.io.PrintWriter;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.misc.InputMappedClassifier;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.Instance;
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
	private InputMappedClassifier imc; 
	
	public Titanic( final String train ) throws Exception 
	{
		train( train ); 
	}
	
	/**
	 * Train a model
	 * @param train
	 * @throws Exception
	 */
	private void train( final String train ) throws Exception
	{ 
		//Load data 
		final DataSource source = new DataSource( train );
		Instances data = source.getDataSet();
		
		
		final String[] nominalClasses = new String[] {
				"Survived","Pclass","Surname","Title","Sex","TicketId","CabinDeck","Embarked"
		};
		data = InstancesManipulation.makeNominal( data, nominalClasses );
		
		
		InstancesManipulation.setClassAttribute( data, "Survived" );
		data = InstancesManipulation.removeAttribute(data, "PassengerId" ); 
		
		
		
		// Subset data to the minimal error model 
		Instances dataSubset = ClassifierCharacterization.getBestAttributeCombination( new SMO(), data, true );
		
		
		/*
		final Evaluation evalJ48 = new Evaluation( data ); 
		evalJ48.crossValidateModel( new J48(), data, 100, new Random() );
		System.out.println( "J48 error: " + evalJ48.errorRate() );
		
		final Evaluation evalNB = new Evaluation( data ); 
		evalNB.crossValidateModel( new NaiveBayes(), data, 100, new Random() );
		System.out.println( "NB error: " + evalNB.errorRate() );
		
		final Evaluation evalIBk = new Evaluation( data ); 
		evalIBk.crossValidateModel( new IBk(), data, 100, new Random() );
		System.out.println( "IBk error: " + evalIBk.errorRate() );
		
		final Evaluation evalOneR = new Evaluation( data ); 
		evalOneR.crossValidateModel( new OneR(), data, 100, new Random() );
		System.out.println( "OneR error: " +  evalOneR.errorRate() );
		
		final Evaluation evalAdaBoostM1 = new Evaluation( data ); 
		evalAdaBoostM1.crossValidateModel( new AdaBoostM1(), data, 100, new Random() );
		System.out.println( "AdaBoostM1 error: " + evalAdaBoostM1.errorRate() );
		
		final Evaluation evalLogitBoost = new Evaluation( data ); 
		evalLogitBoost.crossValidateModel( new LogitBoost(), data, 100, new Random() );
		System.out.println( "LogitBoost error: " + evalLogitBoost.errorRate() );
		
		final Evaluation evalDecisionStump = new Evaluation( data ); 
		evalDecisionStump.crossValidateModel( new DecisionStump(), data, 100, new Random() );
		System.out.println( "DecisionStump error: " + evalDecisionStump.errorRate() );
		
		
		final Evaluation evalLogistic = new Evaluation( data ); 
		evalLogistic.crossValidateModel( new Logistic(), data, 100, new Random() );
		System.out.println( "Logistic error: " +  evalLogistic.errorRate() );
		
		final Evaluation evalSMO = new Evaluation( data ); 
		evalSMO.crossValidateModel( new SMO(), data, 100, new Random() );
		System.out.println( "SMO error: " +  evalSMO.errorRate() );
		*/
		
		/*
		//Train decision tree with all the data
		final J48 j48 = new J48();  
		j48.buildClassifier( data );
		setClassifier( j48 );	
		*/
		
		
		final Evaluation evalSMO = new Evaluation( dataSubset ); 
		evalSMO.crossValidateModel( new SMO(), dataSubset, 100, new Random() );
		System.out.println( "SMO error: " +  evalSMO.errorRate() );
		
		
		/*
		//Train a support vector machine 
		final SMO smo = new SMO(); 
		smo.buildClassifier( dataSubset );
		setClassifier( smo );
		*/
		
		//System.out.println( getClassifier().toString() ); 
		
		
		
		// Subset data to the minimal error model 
		//Instances dataSubset = ClassifierCharacterization.getBestAttributeCombination( new J48(), data, true );
		//Instances dataSubset = data; 
		
		/*
		final J48 j48 = new J48(); 
		j48.buildClassifier( data );
		setClassifier( j48 );
		*/
		
		
		final InputMappedClassifier imc = new InputMappedClassifier(); 
		imc.buildClassifier( data );
		setInputMappedClassifier( imc );
		 
		
	}
	
	public void test( final String testSet, final String predictOut ) throws Exception
	{
		//Load data 
		final DataSource source = new DataSource( testSet );
		Instances data = source.getDataSet();
		
		final String[] nominalClasses = new String[] {
				"Pclass","Surname","Title","Sex","TicketId","CabinDeck","Embarked"
		};
		data = InstancesManipulation.makeNominal( data, nominalClasses ); 
		
		PrintWriter out = TextFileAccess.openFileWrite( predictOut ); 
		out.println( "PassengerId,Survived" ); 
		Instance currentInstance; 
		for( Instance i : data ) 
		{
			currentInstance = getInputMappedClassifier().constructMappedInstance( i );
			final double survived = getClassifier().classifyInstance( currentInstance );
			final String id = i.toString( 0 ); 
			out.println( id + "," + (int) survived ); 
		}
		out.close();   	
	}
	
	private void setClassifier( final AbstractClassifier classifier ) { this.classifier = classifier; }
	public AbstractClassifier getClassifier() { return this.classifier; }
	
	private void setInputMappedClassifier( final InputMappedClassifier imc ) { this.imc = imc; }
	public InputMappedClassifier getInputMappedClassifier() { return this.imc; }
	
	public static void main(String[] args) throws Exception
	{
		final String trainingSet = "resources/titanic/trainClean.csv";
		final String testSet = "resources/titanic/testClean.csv";
		
		
		//for( int i = 1; i < 11; i++ ) {
			final String predictOut = "resources/titanic/predict.csv";
			Titanic tt = new Titanic( trainingSet );
			tt.test( testSet, predictOut );
		//}
		
		
		
		
	
	}

}
