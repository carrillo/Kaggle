package afSIS;

import inputOutput.TextFileAccess;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import neurophTools.DataSetTools;
import neurophTools.MultiLayerPerceptonTools;
import neurophTools.SimpleLearningEventListener;

import org.neuroph.contrib.crossvalidation.CrossValidation;
import org.neuroph.contrib.graphml.GraphmlExport;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.ResilientPropagation;
import org.neuroph.util.TransferFunctionType;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.lazy.IBk;
import weka.classifiers.misc.InputMappedClassifier;
import weka.classifiers.trees.M5P;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;
import wekaTools.InstancesManipulation;
import wekaTools.WekaToNeurophTools;

public class PH1stPrediction {
	
	private Instances trainingData; 
	private Instances testData; 
	
	private ArrayList<String> testIds;
	private ArrayList<String> trainingIds;
	
	private NeuralNetwork ann; 
	
	private AbstractClassifier classifier; 
	private InputMappedClassifier imc; 
	
	public PH1stPrediction( final String trainFileName, final String testFileName ) throws Exception {
		
		//load data 
		loadData( trainFileName, testFileName ); 
		
		preprocessData();  
	}
	
	/**
	 * Load Data 
	 * @param trainFileName
	 * @param testFileName
	 * @throws Exception
	 */
	private void loadData( final String trainFileName, final String testFileName ) throws Exception {
		System.out.println( "Loading training data from file: " + trainFileName ); 
		
		//Load training and testing data 
		setTrainingData( new DataSource( trainFileName ).getDataSet() );
		setTestData( new DataSource( testFileName ).getDataSet() );
		
		//Keep test ids field separately to allow removal of feature 
		final int testN = InstancesManipulation.getFeatureIndex( getTestData(), "id" );
		ArrayList<String> testIds = new ArrayList<String>(); 
		for( int m = 0; m < getTestData().numInstances(); m++ )
		{
			testIds.add( getTestData().instance( m ).toString( testN ) ); 
		}
				
		setTestIds( testIds );
		
		//Keep training ids field separately to allow removal of feature
		final int trainingN = InstancesManipulation.getFeatureIndex( getTrainingData(), "id" );
		ArrayList<String> trainingIds = new ArrayList<String>(); 
		for( int m = 0; m < getTrainingData().numInstances(); m++ )
		{
			trainingIds.add( getTrainingData().instance( m ).toString( trainingN ) ); 
		}
		
		setTrainingIds( trainingIds ); 
	}
	
	/**
	 * Pre-process the training and test data.
	 * 
	 * 1.  
	 */
	private void preprocessData() throws Exception {
		
		
		// Set class attribute to feature "Survived" 
		InstancesManipulation.setClassAttribute( getTrainingData(), "pH" );
				
		//remove unwanted features
		removeFeatures( new String[]{ "id","SOC","Ptransformed","Sand","P", "Ca" }, new String[]{ "id" } );
		
		
	}
	
	/*
	 * Remove unwanted features. 
	 */
	private void removeFeatures( final String[] classesToRemoveTraining, final String[] classesToRemoveTest ) throws Exception
	{

		for( String feature : classesToRemoveTraining )
		{
			setTrainingData( InstancesManipulation.removeAttribute( getTrainingData(), feature ) );
		}
		
		for( String feature : classesToRemoveTest )
		{
			setTestData( InstancesManipulation.removeAttribute( getTestData(), feature ) );
		} 
	}
	
	/*
	 * Train classifier. 
	 * 1. Train classifier
	 * 2. Train InputMappedClassifier to match attribute labels across training and test data. 
	 */
	private void train() throws Exception
	{ 
		System.out.println( "Training model.");
	 
		runEvaluation( new IBk( 1 ), "nearest neighbor 1");
		runEvaluation( new IBk( 2 ), "nearest neighbor 2");
		runEvaluation( new IBk( 4 ), "nearest neighbor 4");
		runEvaluation( new IBk( 8 ), "nearest neighbor 8");
		runEvaluation( new IBk( 16 ), "nearest neighbor 16");
		runEvaluation( new M5P(), "Regression Tree");
		runEvaluation( new LinearRegression(), "Linear Regression");
		//runEvaluation( new MultilayerPerceptron(), "MLP");
		runEvaluation( new SMOreg(), "SMO");
		
		
		
		/*
		// Train classifier (a support vector machine) 
		final SMO smo = new SMO();
		smo.buildClassifier( getTrainingData() );
		setClassifier( smo );
		
		//Map
		final InputMappedClassifier imc = new InputMappedClassifier(); 
		imc.buildClassifier( getTrainingData() );
		setInputMappedClassifier( imc );
		*/
		
		System.out.println( "Training model. Done.");
	}
	
	private void neurophTrain() throws Exception {
		ArrayList<double[]> outputValues =  WekaToNeurophTools.getOuput( getTrainingData() );
		final ArrayList<String> outputLabels = WekaToNeurophTools.getOutputLabels( getTrainingData() ); 
		
		
		final ArrayList<double[]> inputValues = WekaToNeurophTools.getNumericInput( getTrainingData() );
		final ArrayList<String> inputLabels = WekaToNeurophTools.getNumericInputLabels( getTrainingData() );
		
		final DataSet trainingSet = DataSetTools.getTrainingSet( inputValues, outputValues );
		
		
		
		ResilientPropagation resilientBackPropagation = new ResilientPropagation(); 
		resilientBackPropagation.setIncreaseFactor( 1.2 );
		resilientBackPropagation.setDecreaseFactor( 0.5 ) ;
		resilientBackPropagation.setMaxIterations( 50000 );
		
		SimpleLearningEventListener listener = new SimpleLearningEventListener("output/AfSIS/learningProgress.txt", false ); 
		resilientBackPropagation.addListener( listener );
		listener.setVerbose( true );
		
		
		MultiLayerPerceptron myMlPerceptron = new MultiLayerPerceptron( TransferFunctionType.LINEAR, trainingSet.getInputSize(), 1, trainingSet.getOutputSize() ) ;
		MultiLayerPerceptonTools.labelInputOutputNeurons( myMlPerceptron, inputLabels, outputLabels);
		myMlPerceptron.setLearningRule( resilientBackPropagation );

		
		
		myMlPerceptron.learn( trainingSet, resilientBackPropagation );
		setAnn( myMlPerceptron );
	
	}
	
	private void runEvaluation( final AbstractClassifier classifier, final String id ) throws Exception {
		Evaluation evalSMO = new Evaluation( getTrainingData() ); 
		evalSMO.crossValidateModel( classifier, getTrainingData(), 10, new Random( 1000 ) );
		
		System.out.println( id + " error: " +  evalSMO.errorRate() );
	}
	
	public void regressionNeurophTrain( final String predictOut ) throws Exception
	{		
		writePrediction( predictOut, false );
	}
	public void regressionNeurophTest( final String predictOut ) throws Exception
	{		
		 writePrediction( predictOut, true );
	}
	
	
	/*
	 * Classify Instances
	 * 1. Open Output stream 
	 * 2. Classify Instances and write to file. 
	 * 3. Close Output stream. 
	 */
	public void writePrediction( final String fileName, final boolean test ) {
		DataSet testSet = null; 
		ArrayList<String> ids; 
		
		if( test ) {
			testSet = DataSetTools.getTestSet( WekaToNeurophTools.getNumericInput( getTestData() ) );
			ids = getTestIds(); 
		} else {
			testSet = DataSetTools.getTestSet( WekaToNeurophTools.getNumericInput( getTrainingData() ) ); 
			ids = getTrainingIds(); 
		}
		
		System.out.println( "Predict outcome.");
		PrintWriter out = TextFileAccess.openFileWrite( fileName ); 
		out.println( "id,pH" ); 
		
		double prediction;
		for( int m = 0; m < testSet.size(); m++  ) 
		{
			getAnn().setInput( testSet.getRowAt( m ).getInput() );
			getAnn().calculate(); 
			prediction = getAnn().getOutput().clone()[ 0 ];  
			
			out.println( ids.get( m ) + "," + prediction ); 
		}
		out.flush();
		out.close();
		System.out.println( "Predict outcome. Done.");
	}
	
	//Getter and Setter
	private void setTrainingData( final Instances trainingData ) { this.trainingData = trainingData; } 
	public Instances getTrainingData() { return this.trainingData; }
	
	private void setTestData( final Instances testData ) { this.testData = testData; }
	public Instances getTestData() { return this.testData; }
	
	private void setTestIds( final ArrayList<String> ids ) { this.testIds = ids; }
	public ArrayList<String> getTestIds() { return this.testIds; }
	
	private void setTrainingIds( final ArrayList<String> ids ) { this.trainingIds = ids; }
	public ArrayList<String> getTrainingIds() { return this.trainingIds; }
	
	private void setClassifier( final AbstractClassifier classifier ) { this.classifier = classifier; }
	public AbstractClassifier getClassifier() { return this.classifier; }
	
	private void setInputMappedClassifier( final InputMappedClassifier imc ) { this.imc = imc; }
	public InputMappedClassifier getInputMappedClassifier() { return this.imc; }
	
	private void setAnn( final NeuralNetwork ann ) { this.ann = ann; } 
	public NeuralNetwork getAnn() { return this.ann; } 
	
	
	/*
	 * Predict soil paramtersts. 
	 * 1. Define input data (training and test data). 
	 * 2. Define output data (predictions) 
	 *  
	 */
	public static void main(String[] args) throws Exception
	{
		final long time = System.currentTimeMillis();
		
		//Specify data 
		final String trainingSet = "resources/AfSIS/trainingTransformed.csv";
		
		//final String trainingOriginalSet = "resources/AfSIS/trainingOriginal.csv";
		
		final String testSet = "resources/AfSIS/testTransformed.csv";
		
		final String predictOut = "resources/AfSIS/predict_pH.csv";
		
		PH1stPrediction afsis = new PH1stPrediction( trainingSet, testSet );
		
		//afsis.train();
		afsis.neurophTrain(); 
		
		afsis.regressionNeurophTest( "resources/AfSIS/predict_pH.csv" );
		afsis.regressionNeurophTrain( "resources/AfSIS/predictTrain_pH.csv" );
		
			 
		System.out.println( "Done. [" + (System.currentTimeMillis() - time)/1000 + " s]" );

	}

}
