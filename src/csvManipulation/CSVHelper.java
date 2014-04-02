package csvManipulation;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

public class CSVHelper 
{
	/*
	 * Read CSV file using standard settings. 
	 */
	public static Iterable<CSVRecord> readCSV( final String fileName ) throws IOException
	{
		Reader in = new FileReader( fileName );
		Iterable<CSVRecord> records = CSVFormat.DEFAULT.parse( in );
		return records; 
	}
}
