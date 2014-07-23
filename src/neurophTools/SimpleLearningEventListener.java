package neurophTools;

import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.learning.SupervisedLearning;
import org.neuroph.core.events.LearningEventListener;

public class SimpleLearningEventListener implements LearningEventListener 
{
	private Long timeOfLastEvent = null; 
	
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
		
		System.out.println( "Current iteration: " + sl.getCurrentIteration() + ". Duration: " + time + "ms." + " Previous epoch error: " + sl.getPreviousEpochError() );
		
	}

}
