package titanic;

import inputOutput.TextFileAccess;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;

import libsvmTools.ClassifierParameters;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.misc.InputMappedClassifier;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Add;
import wekaTools.InstancesManipulation;
import wekaTools.MatchDataSets;
import wekaTools.NormalizeNumericFeatures;

/**
 * Predicting the survival probability of titanic passengers. 
 * The data set is provided by kaggle. 
 * Try different classification methods on the data. 
 * @author carrillo
 *
 */
public class TitanicTesting 
{ 
	private AbstractClassifier classifier;
	
	private InputMappedClassifier imc; 
	private Instances trainingData, testData;
	private ArrayList<String> testIds; 
	
	public TitanicTesting( final String trainFileName, final String testFileName ) throws Exception 
	{
		preprocessData( trainFileName, testFileName );
	}
	
	/*
	 * Pre-process the training and test data. 
	 * 
	 * 1. Load the data. 
	 * 2. Make nominal features nominal.
	 * 3. Match training and test data (Add missing features and missing labels in nominal features). 
	 * 3. Define class feature ("Survived") in data.  
	 * 4. Keep only useful features identified during data exploration. 
	 * 5. Set Training and Test data instance variables. 
	 */
	public void preprocessData( final String trainFileName, final String testFileName ) throws Exception 
	{
		System.out.println( "Pre-processing data."); 
		// Load training and test data 
		setTrainingData( new DataSource( trainFileName ).getDataSet() );
		setTestData( new DataSource( testFileName ).getDataSet() );
				
		// Add survived feature (with empty values) to test data to make it compatible and keep the id field 
		keepPassengerIdOfTestData();
		
		// Make nominal features nominal 
		defineNominalClasses();
		
		// Match test and training data  
		MatchDataSets matchDataSets = new MatchDataSets( getTrainingData(), getTestData() );
		matchDataSets.run(); 
		setTrainingData( matchDataSets.getInstance1() );
		setTestData( matchDataSets.getInstance2() ); 
		
		// Set class attribute to feature "Survived" 
		InstancesManipulation.setClassAttribute( getTrainingData(), "Survived" ); 
		
		// Normalize features to range between 0 and 1. 
		normalizeNumericalFeatures();		
		
		// Remove unwanted features
		removeFeatures();  
		
		// Convert nominal features with k-values into k features with binary values. 
		nominalToBinary();
		
		System.out.println( "Pre-processing data. Done.");
	}
	
	/*
	 * Prepare the test data for processing 
	 * 1. Add the survived attribute to test data.
	 * 2. Keep the passengerId field for identification of prediction.  
	 */
	private void keepPassengerIdOfTestData() throws Exception 
	{
		//Keep PassengerId
		final int n = InstancesManipulation.getFeatureIndex( getTestData(), "PassengerId" );
		ArrayList<String> ids = new ArrayList<String>(); 
		for( int m = 0; m < getTestData().numInstances(); m++ )
		{
			ids.add( getTestData().instance( m ).toString( n ) ); 
		}
		
		setTestIds( ids );	
	}
	
	/*
	 * Define which features are nominal features 
	 */
	private void defineNominalClasses() throws Exception
	{
		final String[] nominalClasses = new String[] {
				"Survived"/*,"Pclass"*/,"Surname","Title","Sex","TicketId","CabinDeck","Embarked"
		};
		setTrainingData( InstancesManipulation.makeNominal( getTrainingData(), nominalClasses ) );
		setTestData( InstancesManipulation.makeNominal( getTestData(), nominalClasses ) );
	}
	
	/*
	 * Remove unwanted features. 
	 */
	private void removeFeatures() throws Exception
	{
		//Remove features
		final String[] classesToRemove = new String[] {
				//"Surname","TicketId","Cabin", //Nominal Features
				"TicketId", //Nominal Features
				"PassengerId","CabinCount","CabinNr" //Numerical Features
		};
		
		for( String feature : classesToRemove )
		{
			setTrainingData( InstancesManipulation.removeAttribute( getTrainingData(), feature ) ); 
		}  
	}
	
	/*
	 * Scales numerical features between 0 and 1.
	 * 1. Separate class feature from rest
	 * 2. Train normalization on training data.
	 * 3. Scale training data. 
	 * 4. Join class feature with scaled training data.
	 *     
	 * 5. Scale test data    
	 */
	private void normalizeNumericalFeatures() throws Exception
	{
		final NormalizeNumericFeatures norm = new NormalizeNumericFeatures(); 
		norm.setInputFormat( getTrainingData() );
		
		setTrainingData( Filter.useFilter( getTrainingData(), norm ) ) ;
		setTestData( Filter.useFilter( getTestData(), norm ) );
	}
	
	private void nominalToBinary() throws Exception
	{
		NominalToBinary nominal2Binary = new NominalToBinary(); 
		nominal2Binary.setInputFormat( getTrainingData() );
		
		setTrainingData( Filter.useFilter( getTrainingData(), nominal2Binary ) );
		setTestData( Filter.useFilter( getTestData(), nominal2Binary ) );
	}
	
	/*
	 * Train a model
	 */
	private void train() throws Exception
	{ 
		System.out.println( "Training model.");
	 
		
		//final int kFoldXValidation = getTrainingData().numInstances();
		
		
		final int kFoldXValidation = 100;
		
		/*
		final Evaluation evalJ48 = new Evaluation( getTrainingData() ); 
		evalJ48.crossValidateModel( new J48(), getTrainingData(), kFoldXValidation, new Random() );
		System.out.println( "J48 error: " + evalJ48.errorRate() );
		System.out.println( "J48 kappa: " + evalJ48.kappa() );
		
		final Evaluation evalNB = new Evaluation( getTrainingData() ); 
		evalNB.crossValidateModel( new NaiveBayes(), getTrainingData(), kFoldXValidation, new Random() );
		System.out.println( "NB error: " + evalNB.errorRate() );
		System.out.println( "NB kappa: " + evalNB.kappa() );
		
		final Evaluation evalIBk = new Evaluation( getTrainingData() ); 
		evalIBk.crossValidateModel( new IBk(), getTrainingData(), kFoldXValidation, new Random() );
		System.out.println( "IBk error: " + evalIBk.errorRate() );
		System.out.println( "IBk kappa: " + evalIBk.kappa() );
		
		final Evaluation evalOneR = new Evaluation( getTrainingData() ); 
		evalOneR.crossValidateModel( new OneR(), getTrainingData(), kFoldXValidation, new Random() );
		System.out.println( "OneR error: " +  evalOneR.errorRate() );
		System.out.println( "OneR kappa: " + evalOneR.kappa() );
		
		final Evaluation evalAdaBoostM1 = new Evaluation( getTrainingData() ); 
		evalAdaBoostM1.crossValidateModel( new AdaBoostM1(), getTrainingData(), kFoldXValidation, new Random() );
		System.out.println( "AdaBoostM1 error: " + evalAdaBoostM1.errorRate() );
		System.out.println( "AdaBoostM1 kappa: " + evalAdaBoostM1.kappa() );
		
		final Evaluation evalLogitBoost = new Evaluation( getTrainingData() ); 
		evalLogitBoost.crossValidateModel( new LogitBoost(), getTrainingData(), kFoldXValidation, new Random() );
		System.out.println( "LogitBoost error: " + evalLogitBoost.errorRate() );
		System.out.println( "LogitBoost kappa: " + evalLogitBoost.kappa() );
		
		final Evaluation evalDecisionStump = new Evaluation( getTrainingData() ); 
		evalDecisionStump.crossValidateModel( new DecisionStump(), getTrainingData(), kFoldXValidation, new Random() );
		System.out.println( "DecisionStump error: " + evalDecisionStump.errorRate() );
		System.out.println( "DecisionStump kappa: " + evalDecisionStump.kappa() );
		
		
		final Evaluation evalLogistic = new Evaluation( getTrainingData() ); 
		evalLogistic.crossValidateModel( new Logistic(), getTrainingData(), kFoldXValidation, new Random() );
		System.out.println( "Logistic error: " +  evalLogistic.errorRate() );
		System.out.println( "Logistic kappa: " +  evalLogistic.kappa() );
		
		
		
		
		Evaluation evalSMO = new Evaluation( getTrainingData() ); 
		evalSMO.crossValidateModel( new SMO(), getTrainingData(), kFoldXValidation, new Random( 1000 ) );
		System.out.println( "SMO error: " +  evalSMO.errorRate() );
		System.out.println( "SMO kappa: " +  evalSMO.kappa() );
		*/
		
		
		
		
		// setup classifier
	    CVParameterSelection ps = new CVParameterSelection();
	    
	    ps.setClassifier( new SMO() );
	    ps.setNumFolds(5);  // using 5-fold CV
	    ps.addCVParameter("C 0.92 0.93 10");
	    

	    // build and output best options
	    ps.buildClassifier( getTrainingData() );
	    System.out.println( ps.toSummaryString() );
	    
	    final Evaluation evalSMO = new Evaluation( getTrainingData() ); 
		evalSMO.crossValidateModel( ps.getClassifier(), getTrainingData(), 100, new Random( 1000 ) );
		System.out.println( "SMO error: " +  evalSMO.errorRate() );
		System.out.println( "SMO kappa: " +  evalSMO.kappa() );
		
		
		
		/*
		final ClassifierParameters libsvmParameter = new ClassifierParameters( new LibSVM() );
		
		libsvmParameter.setValue( "-E" , 0.001 );
		final ArrayList<Evaluation> evaluations = new ArrayList<Evaluation>(); 
		final double[] error = new double[ 4 ]; 
		for( int i = 0; i < 4; i++ )
		{
			final Evaluation evalSVM = new Evaluation( getTrainingData() );  
			LibSVM svmTest = new LibSVM();
			
			libsvmParameter.setValue( "-K", i ); 
			 
			svmTest.setOptions( libsvmParameter.getLibSVMOptionStringArray() );
			
			//System.out.println( StringArrayTools.arrayToString(svmTest.getOptions(), ",") ); 
			  
			evalSVM.crossValidateModel( svmTest, getTrainingData(), 10, new Random( 1000 ) );
			evaluations.add( evalSVM ); 
			error[ i ] = evalSVM.errorRate(); 
		}
		
	
		//Print error 
		for( Evaluation eval : evaluations )
		{
			System.out.println( "Error " + eval.errorRate() );
			System.out.println( "Kappa " + eval.kappa() ); 
		}
		
		*/
		
		
		// Train classifier (a support vector machine) 
		final SMO smo = new SMO();
		//smo.setOptions( ps.getBestClassifierOptions() );
		smo.buildClassifier( getTrainingData() );
		setClassifier( smo );
		
		
		final InputMappedClassifier imc = new InputMappedClassifier(); 
		imc.buildClassifier( getTrainingData() );
		setInputMappedClassifier( imc );
		
		
		System.out.println( "Training model. Done.");
	}
	
	/*
	 * Classify Instances
	 * 1. Open Output stream 
	 * 2. Classify Instances and write to file. 
	 * 3. Close Output stream. 
	 */
	public void classify( final String predictOut ) throws Exception
	{		
		System.out.println( "Predict outcome.");
		PrintWriter out = TextFileAccess.openFileWrite( predictOut ); 
		out.println( "PassengerId,Survived" ); 
		Instance currentInstance;  
		for( int m = 0; m < getTestData().numInstances(); m++  ) 
		{
			final String id = getTestIds().get( m );  
			currentInstance = getInputMappedClassifier().constructMappedInstance( getTestData().get( m ) );
			 
			final double survived = getClassifier().classifyInstance( currentInstance );
			out.println( id + "," + (int) survived ); 
		}
		out.flush();
		out.close();
		System.out.println( "Predict outcome. Done.");
	}
	
	private void setClassifier( final AbstractClassifier classifier ) { this.classifier = classifier; }
	public AbstractClassifier getClassifier() { return this.classifier; }
	
	private void setInputMappedClassifier( final InputMappedClassifier imc ) { this.imc = imc; }
	public InputMappedClassifier getInputMappedClassifier() { return this.imc; }
	
	private void setTrainingData( final Instances trainingData ) { this.trainingData = trainingData; } 
	public Instances getTrainingData() { return this.trainingData; }
	
	private void setTestData( final Instances testData ) { this.testData = testData; }
	public Instances getTestData() { return this.testData; }
	
	private void setTestIds( final ArrayList<String> testIds ) { this.testIds = testIds; }
	public ArrayList<String> getTestIds() { return this.testIds; } 
	
	public static void main(String[] args) throws Exception
	{
		final long time = System.currentTimeMillis();
		
		final String trainingSet = "resources/titanic/trainClean.csv";
		final String testSet = "resources/titanic/testClean.csv";
		
		final String predictOut = "resources/titanic/predict.csv";
		TitanicTesting tt = new TitanicTesting( trainingSet, testSet );
		tt.train(); 
		tt.classify( predictOut );
			 
			
		System.out.println( "Done. [" + (System.currentTimeMillis() - time)/1000 + " s]" );
	}

}
