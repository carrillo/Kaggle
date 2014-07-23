package neurophTools;

import java.util.ArrayList;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

import array.tools.DoubleArrayTools;

public class DataSetTools 
{
	public static DataSet getTrainingSet( final ArrayList<double[]> inputValues, final ArrayList<double[]> outputValues )
	{
		DataSet trainingSet = new DataSet( inputValues.get( 0 ).length, outputValues.get( 0 ).length ); 
		for( int i = 0; i < inputValues.size(); i++ )
		{  
			trainingSet.addRow( new DataSetRow( inputValues.get( i ), outputValues.get( i ) ) );
		}
		return trainingSet; 
	}


}
