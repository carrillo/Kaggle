package afSIS;

import java.util.ArrayList;
import java.util.Random;

import neurophTools.DataSetTools;
import neurophTools.MultiLayerPerceptonTools;
import neurophTools.SimpleLearningEventListener;

import org.neuroph.contrib.crossvalidation.CrossValidation;
import org.neuroph.core.data.DataSet;
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
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;
import wekaTools.InstancesManipulation;
import wekaTools.WekaToNeurophTools;

public class AfSIS {
	
	private Instances trainingData; 
	private Instances testData; 
	
	private ArrayList<String> testIds;
	
	private AbstractClassifier classifier; 
	private InputMappedClassifier imc; 
	
	public AfSIS( final String trainFileName, final String testFileName ) throws Exception {
		
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
		
		//Keep training ids field separately to allow removal of feature 
		final int n = InstancesManipulation.getFeatureIndex( getTestData(), "id" );
		ArrayList<String> ids = new ArrayList<String>(); 
		for( int m = 0; m < getTestData().numInstances(); m++ )
		{
			ids.add( getTestData().instance( m ).toString( n ) ); 
		}
				
		setTestIds( ids );
	}
	
	/**
	 * Pre-process the training and test data.
	 * 
	 * 1.  
	 */
	private void preprocessData() throws Exception {
		
		
		// Set class attribute to feature "Survived" 
		InstancesManipulation.setClassAttribute( getTrainingData(), "P" );
				
		//remove unwanted features
		removeFeatures( new String[]{ "id","SOC","Ptransformed","pH","Sand", "Ca" } );
		
		
	}
	
	/*
	 * Remove unwanted features. 
	 */
	private void removeFeatures( final String[] classesToRemove ) throws Exception
	{

		for( String feature : classesToRemove )
		{
			setTrainingData( InstancesManipulation.removeAttribute( getTrainingData(), feature ) ); 
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
		resilientBackPropagation.setMaxIterations( 100 );
		
		SimpleLearningEventListener listener = new SimpleLearningEventListener("output/AfSIS/learningProgress.txt", false ); 
		resilientBackPropagation.addListener( listener );
		listener.setVerbose( true );
		
		
		MultiLayerPerceptron myMlPerceptron = new MultiLayerPerceptron( TransferFunctionType.LINEAR, trainingSet.getInputSize(), 1, trainingSet.getOutputSize() ) ;
		MultiLayerPerceptonTools.labelInputOutputNeurons( myMlPerceptron, inputLabels, outputLabels);
		myMlPerceptron.setLearningRule( resilientBackPropagation );

		
		CrossValidation cv = new CrossValidation( trainingSet, myMlPerceptron, trainingSet.size() ); 
		cv.run();
		cv.writeAllObservedVsPredictedPairs( "output/AfSIS/P_XValidation.csv" );
		
		 
		
		/*
		HiddenNodeLearningRule dnlr = new HiddenNodeLearningRule(trainingSet, 5, TransferFunctionType.TANH, 
				resilientBackPropagation, "output/AfSIS/learningCurveHiddenNodes_SOC.txt" );
		
		dnlr.run( 0, 50, 1 );
		
		listener.setVerbose( true );
		resilientBackPropagation.setMaxIterations( 10000 );
		
		
		
		//myMlPerceptron.learn( trainingSet, resilientBackPropagation );
		
		GraphmlExport graphmlExport = new GraphmlExport( myMlPerceptron ); 
		graphmlExport.parse();
		//graphmlExport.writeToFile( new String("output/AfSIS/SOC.graphml") );
		*/
	}
	
	private void runEvaluation( final AbstractClassifier classifier, final String id ) throws Exception {
		Evaluation evalSMO = new Evaluation( getTrainingData() ); 
		evalSMO.crossValidateModel( classifier, getTrainingData(), 10, new Random( 1000 ) );
		
		System.out.println( id + " error: " +  evalSMO.errorRate() );
	}

	//Getter and Setter
	private void setTrainingData( final Instances trainingData ) { this.trainingData = trainingData; } 
	public Instances getTrainingData() { return this.trainingData; }
	
	private void setTestData( final Instances testData ) { this.testData = testData; }
	public Instances getTestData() { return this.testData; }
	
	private void setTestIds( final ArrayList<String> ids ) { this.testIds = ids; }
	public ArrayList<String> getTestIds() { return this.testIds; }
	
	private void setClassifier( final AbstractClassifier classifier ) { this.classifier = classifier; }
	public AbstractClassifier getClassifier() { return this.classifier; }
	
	private void setInputMappedClassifier( final InputMappedClassifier imc ) { this.imc = imc; }
	public InputMappedClassifier getInputMappedClassifier() { return this.imc; }
	
	
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
		
		final String trainingSetP90Quantile = "resources/AfSIS/trainingTransformedP90Quantile.csv";
		final String trainingSetP95Quantile = "resources/AfSIS/trainingTransformedP95Quantile.csv";
		final String trainingSetP99Quantile = "resources/AfSIS/trainingTransformedP99Quantile.csv";
		
		final String trainingOriginalSet = "resources/AfSIS/trainingOriginal.csv";
		
		final String testSet = "resources/AfSIS/testTransformed.csv";
		
		final String predictOut = "resources/AfSIS/predict.csv";
		
		AfSIS afsis = new AfSIS( trainingSet, testSet );
		//AfSIS afsis = new AfSIS( trainingOriginalSet, testSet );
		
		//afsis.train();
		afsis.neurophTrain(); 
		
		
			 
		System.out.println( "Done. [" + (System.currentTimeMillis() - time)/1000 + " s]" );

	}

}
