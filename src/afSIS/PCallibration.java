package afSIS;

import inputOutput.TextFileAccess;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import neurophTools.DataSetTools;
import neurophTools.GreedyFeatureSelection;
import neurophTools.MultiLayerPerceptonTools;
import neurophTools.SimpleLearningEventListener;

import org.neuroph.contrib.crossvalidation.CrossValidation;
import org.neuroph.contrib.crossvalidation.HiddenNodeLearningRule;
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
import wekaTools.AttributeTransformation;
import wekaTools.InstancesManipulation;
import wekaTools.WekaToNeurophTools;

public class PCallibration {
	
	private Instances trainingData; 
	private Instances testData; 
	
	private ArrayList<String> testIds;
	private ArrayList<String> trainingIds;
	
	private NeuralNetwork ann; 
	
	private AbstractClassifier classifier; 
	private InputMappedClassifier imc; 
	
	public PCallibration( final String trainFileName, final String testFileName ) throws Exception {
		
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
		InstancesManipulation.setClassAttribute( getTrainingData(), "P" );
				
		//remove unwanted features
		removeFeatures( new String[]{ "id","pH","Sand","SOC", "Ca" }, new String[]{ "id" } );
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
	
	
	private void addPolynomialTerms() throws Exception
	{
		System.out.println( "Adding polynomial expansion of features. Before: " + getTrainingData().numAttributes() + "\t" + getTestData().numAttributes() ); 
		
		setTrainingData( AttributeTransformation.addPolynomialCombinations( getTrainingData() ) );
		setTestData( AttributeTransformation.addPolynomialCombinations( getTestData() ) );
		
		System.out.println( "Adding polynomial expansion of features. After: " + getTrainingData().numAttributes() + "\t" + getTestData().numAttributes() ); 
	}
			
	/*
	 * Train classifier. 
	 * 1. Train classifier
	 * 2. Train InputMappedClassifier to match attribute labels across training and test data. 
	 */
	private void train() throws Exception
	{ 
		System.out.println( "Training model.");
	 
		
		wekaNearestNeighborEvaluation(); 
		//wekaSmoRegEvaluation();
		wekaRegressionTreeEvaluation(); 
		wekaLinearRegressionEvaluation();
		
		
		
		
				
		
		
		
		
	
		 
		
		
		
		
		
		//
		runEvaluation( new MultilayerPerceptron(), "MLP");
		//runEvaluation( new SMOreg(), "SMO");
		
		
		
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
	
	private void wekaNearestNeighborEvaluation() throws Exception {
		
		AbstractClassifier ibk = new IBk();
		String[] options = new String[ 3 ];
		options[ 0 ] = "-K"; 
		options[ 1 ] = "18";
		options[ 2 ] = "-I";   
		
		ibk.setOptions( options.clone() );
		runEvaluation( ibk, "Nearest neighbor" );
	}
	
	
	private void wekaSmoRegEvaluation() throws Exception {
		
		AbstractClassifier smoReg = new SMOreg();
		System.out.println( Arrays.toString( smoReg.getOptions() ) ); 
		//runEvaluation( smoReg, "SMO Regression");
		
		//for( double i = 1; i < 100; i = ( i * (double) 2 ) ) {
			String[] options = new String[ 21 ];
			options[ 0 ] = "-C"; 
			options[ 1 ] = String.valueOf( 1 ); 
			options[ 2 ] = "-N"; 
			options[ 3 ] = "0"; 
			options[ 4 ] = "-I"; 
			options[ 5 ] = "weka.classifiers.functions.supportVector.RegSMOImproved"; 
			options[ 6 ] = "-L";
			options[ 7 ] = "0.001"; 
			options[ 8 ] = "-W"; 
			options[ 9 ] = "1";
			options[ 10 ] = "-P"; 
			options[ 11 ] = "1.0E-12";
			options[ 12 ] = "-T"; 
			options[ 13 ] = "0.01";
			options[ 14 ] = "-V";
			options[ 15 ] = "-K";

			options[ 16 ] = "weka.classifiers.functions.supportVector.PolyKernel";
			options[ 17 ] = "-C";
			options[ 18 ] = "0";
			options[ 19 ] = "-E";
			options[ 20 ] = "2.0";

			smoReg.setOptions( options.clone() );
			System.out.println( Arrays.toString( smoReg.getOptions() ) );
			runEvaluation( smoReg, "SMO Regression, C parameter: ");
						
			
			options[ 16 ] = "weka.classifiers.functions.supportVector.Puk";
			smoReg.setOptions( options.clone() );
			System.out.println( Arrays.toString( smoReg.getOptions() ) );
			runEvaluation( smoReg, "SMO Regression, C parameter: "); 
			 
		
			/*
			options[ 16 ] = "weka.classifiers.functions.supportVector.RBFKernel";
			smoReg.setOptions( options.clone() );
			System.out.println( Arrays.toString( smoReg.getOptions() ) );
			runEvaluation( smoReg, "SMO Regression, C parameter: ");
			
			options[ 16 ] = "weka.classifiers.functions.supportVector.PolyKernel";
			smoReg.setOptions( options.clone() );
			System.out.println( Arrays.toString( smoReg.getOptions() ) );
			runEvaluation( smoReg, "SMO Regression, C parameter: ");
			*/
			
			
			
		//}
		
		
	}
	
	private void wekaLinearRegressionEvaluation() throws Exception {
		AbstractClassifier linRegression = new LinearRegression();
		runEvaluation( linRegression, "Linear Regression");
		
		String[] options = new String[ 4 ];
		options[ 0 ] = "-S"; 
		options[ 1 ] = "0"; 
		options[ 2 ] = "-R"; 
		options[ 3 ] = "0.87";  
		
		linRegression.setOptions( options );
		runEvaluation( linRegression, "Linear Regression, ridge parameter: " + 0.87 );
	}
	
	private void wekaRegressionTreeEvaluation() throws Exception {
		AbstractClassifier m5p = new M5P();
		 
		String[] options = new String[ 3 ];
		options[0] = "-N";
		options[1] = "-M";
		options[2] = "8"; 
		m5p.setOptions( options );
		runEvaluation( m5p, "Regression Tree " + 8 + " instances per leaf: ");
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
		
		
		MultiLayerPerceptron myMlPerceptron = new MultiLayerPerceptron( TransferFunctionType.LINEAR, trainingSet.getInputSize(), 5, trainingSet.getOutputSize() ) ;
		MultiLayerPerceptonTools.labelInputOutputNeurons( myMlPerceptron, inputLabels, outputLabels);
		myMlPerceptron.setLearningRule( resilientBackPropagation );

		CrossValidation cv = new CrossValidation( trainingSet, myMlPerceptron, trainingSet.size() ); 
		//cv.run();
		cv.writeAllObservedVsPredictedPairs( "output/AfSIS/P_XValidation.csv" );
		
		System.out.println( "all features: " + cv.getMeanError() ); 
		
		listener.setVerbose( false );
		//GreedyFeatureSelection fs = new GreedyFeatureSelection( resilientBackPropagation, getTrainingData() ); 
		//fs.run();
		listener.setVerbose( false );
		
		
		 
		
		
		HiddenNodeLearningRule dnlr = new HiddenNodeLearningRule(trainingSet, trainingSet.size(), TransferFunctionType.LINEAR, 
				resilientBackPropagation, "output/AfSIS/learningCurveHiddenNodes_P.txt" );
		
		dnlr.run( 0, 50, 1 );
		
		listener.setVerbose( true );
		resilientBackPropagation.setMaxIterations( 10000 );
		
		
	
		//myMlPerceptron.learn( trainingSet, resilientBackPropagation );
		//setAnn( myMlPerceptron );
	
	
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
		out.println( "id,P" ); 
		
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
		final String trainingSet = "resources/AfSIS/callibrationTrain.csv";
		final String testSet = "resources/AfSIS/callibrationTest.csv";
		
		
		
		PCallibration afsis = new PCallibration( trainingSet, testSet );
		
		//afsis.addPolynomialTerms();
		afsis.train();

		afsis.neurophTrain(); 
		//afsis.regressionNeurophTest( "resources/AfSIS/callibratedTest_P.csv" );
		//afsis.regressionNeurophTrain( "resources/AfSIS/callibratedTrain_P.csv" );
		
			 
		System.out.println( "Done. [" + (System.currentTimeMillis() - time)/1000 + " s]" );

	}

}
