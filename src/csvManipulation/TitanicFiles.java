package csvManipulation;

import inputOutput.TextFileAccess;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import org.apache.commons.csv.CSVRecord;

public class TitanicFiles 
{
	
	public TitanicFiles( final String fileNameIn, final String fileNameOut, final boolean isTestData ) throws IOException 
	{
		Iterable<CSVRecord> records = CSVHelper.readCSV( fileNameIn );
		PrintWriter out = TextFileAccess.openFileWrite( fileNameOut ); 
		 
		String header = "PassengerId,Survived,Pclass,Surname,Title,Sex,Age,SibSp,Parch,TicketId,TicketNr,Fare,CabinCount,CabinDeck,CabinNr,Embarked";
		if( isTestData )
			header = "PassengerId,Pclass,Surname,Title,Sex,Age,SibSp,Parch,TicketId,TicketNr,Fare,CabinCount,CabinDeck,CabinNr,Embarked";
		
		out.println( header ); 
		
		int count = 0;
		for( CSVRecord record : records )
		{
			if( count > 0 )
			{
				out.println( transformRecord( record, isTestData ) );  
			}
			count++; 
		}
		out.close(); 
	}
	
	/*
	 * Add newly formatted attribute and replace missing values by '?'
	 * 1. Take first entry as they are ("PassengerId","Survived","Pclass")
	 * 2. Split name into "Surname","Title"
	 * 3. Take next 4 entries as they are ("Sex","Age","SibSp","Parch").
	 * 4. Split ticket name into character and numeric value 
	 * 5. Take next entry as they are ("Fare","Cabin","Embarked") 
	 * 6. Split Cabin into character and numeric value
	 */
	private String transformRecord( final CSVRecord csvRecord, final boolean isTestData ) 
	{
		int offSet = 0; 
		if( isTestData )
			offSet = -1; 
		
		//Add first attributes as they are 
		String out = stringToWekaEntry( csvRecord.get( 0 ) ) + ",";
		
		//Add survival attribure only in the train data 
		if( !isTestData )
			out += stringToWekaEntry( csvRecord.get( 1 ) ) + ","; 
		
		//Add remaining data with offsetted index if parsing test data 
		out += stringToWekaEntry(csvRecord.get( 2 + offSet ) ) + ",";
		
		//Add split name attribute
		final String[] splitName = splitName( csvRecord.get( 3 + offSet ) ); 
		out += stringToWekaEntry( splitName[ 0 ] ) + "," + stringToWekaEntry( splitName[ 1 ] ) + ","; 
		
		//Add next attributes as they are
		out += stringToWekaEntry( csvRecord.get( 4 + offSet ) ) + ",";
		out += stringToWekaEntry( csvRecord.get( 5 + offSet ) ) + ",";
		out += stringToWekaEntry( csvRecord.get( 6 + offSet ) ) + ",";
		out += stringToWekaEntry( csvRecord.get( 7 + offSet ) ) + ",";
		
		//Add split ticket attribute
		final String[] splitTicket = splitTicket( csvRecord.get( 8 + offSet ) );
		out += stringToWekaEntry( splitTicket[ 0 ] ) + "," + stringToWekaEntry( splitTicket[ 1 ] ) + ",";  
		
		//Add next attributes as they are 
		out += stringToWekaEntry( csvRecord.get( 9 + offSet ) ) + ","; 
		
		//Add split cabin attribute
		final String[] splitCabin = splitCabin( csvRecord.get( 10 + offSet ) );
		out += stringToWekaEntry( splitCabin[ 0 ] ) + "," + stringToWekaEntry( splitCabin[ 1 ] ) + "," + stringToWekaEntry( splitCabin[ 2 ] ) + ",";  
		
		//Add final attribute as it is 
		out += stringToWekaEntry( csvRecord.get( 11 + offSet ) );  
		
		return out; 
	}
	
	/*
	 *  Generates a Weka entry from a CSVRecord (i.e. inserts ? for missing values). 
	 */
	private String stringToWekaEntry( final String csvRecord ) 
	{
		String out = csvRecord; 
		
		if( out.length() == 0 )
			out += "?"; 
		
		return out; 
	}

	
	/*
	 * Splits name into surname and title 
	 */
	private String[] splitName( final String nameField ) 
	{ 
		final String[] commaSplit = nameField.split(", ");
		
		String surname = commaSplit[ 0 ].trim();
		surname = surname.replace("'", ""); 
		assert ( surname.length() != 0 ) : "No surname found: " + nameField;
		
		final String title = commaSplit[ 1 ].split(". ")[ 0 ]; 
		assert ( title.length() != 0 ) : "No title found: " + nameField;
		
		return new String[]{ surname, title }; 
	}
	
	/*
	 * Split ticket to character field and number field 
	 */
	private String[] splitTicket( final String ticketField )
	{
		
		
		final String[] spaceSplit = ticketField.split( " " ); 
		final String[] out = new String[ 2 ];
		
		if( ticketField.equals( "LINE" ) ) 
		{
			out[ 0 ] = "LINE";
			out[ 1 ] = ""; 
		} 
		else 
		{			
			// last field is ticket number 
			out[ 1 ] = spaceSplit[  spaceSplit.length - 1 ];
			// remaining is interpreted as character
			out[ 0 ] = ticketField.substring(0, ticketField.indexOf( out[ 1 ] ) ); 
		}
		
		out[ 0 ] = out[ 0 ].trim(); 
		out[ 1 ] = out[ 1 ].trim(); 
		
		return out; 
	}
	
	/*
	 * Split cabin field into cabin count, last deck mentioned and last cabin number (mean of many values)  
	 */
	private String[] splitCabin( String cabinField )
	{
		//Initiate empty array 
		final String[] out = new String[ 3 ];
		for( int i = 0; i < out.length; i++ )
			out[ i ] = ""; 
			
			
		final String[] cabins = cabinField.split(" "); 
		
		if( cabinField.length() != 0 ) 
		{
			out[ 0 ] = String.valueOf( cabins.length ); 
			out[ 1 ] = cabins[ cabins.length - 1 ].substring(0,1);
			out[ 2 ] = cabins[ cabins.length - 1 ].substring( 1 );  
		}  
		
		return out; 
	}
	
	public static void main(String[] args) throws IOException
	{
		String fileNameIn = "resources/titanic/test.csv";
		String fileNameOut = "resources/titanic/testClean.csv";
		
		final boolean isTestData = true; 
		
		new TitanicFiles( fileNameIn, fileNameOut, isTestData ); 
	}

}
