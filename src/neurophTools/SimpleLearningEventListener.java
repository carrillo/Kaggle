package neurophTools;

import inputOutput.TextFileAccess;

import java.io.PrintWriter;

import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.learning.SupervisedLearning;
import org.neuroph.core.events.LearningEventListener;

public class SimpleLearningEventListener implements LearningEventListener 
{
	private Long timeOfLastEvent = null;
	private PrintWriter logOut = null; 
	private boolean verbose; 
	
	public SimpleLearningEventListener( final String logFile ) {
		this( logFile, true );  
	}
	
	public SimpleLearningEventListener( final String logFile, final boolean verbose ) {
		this.logOut = TextFileAccess.openFileWrite( logFile );
		this.verbose = verbose; 
		logOut.println( "iteration,error" ); 
	}
	
	public SimpleLearningEventListener(){
		
	}
	
	@Override
	public void handleLearningEvent( LearningEvent arg0 ) {
		 
		Long lastEvent = this.timeOfLastEvent; 
		this.timeOfLastEvent = System.currentTimeMillis();
		
		long time; 
		if( lastEvent == null )
		{
			time = 0; 
		}
		else 
		{
			time = this.timeOfLastEvent - lastEvent; 
		}
		
		SupervisedLearning sl = (SupervisedLearning ) arg0.getSource();
		
		if( verbose )
			System.out.println( "Current iteration: " + sl.getCurrentIteration() + ". Duration: " + time + "ms." + " Previous epoch error: " + sl.getPreviousEpochError() );
		
		if( logOut != null ) {
			logOut.println( sl.getCurrentIteration() + "," + sl.getPreviousEpochError() );
			logOut.flush(); 
		}
		
	}
	
	public void setVerbose( final boolean verbose ) { this.verbose = verbose; } 

}
