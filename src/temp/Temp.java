package temp;

import java.util.ArrayList;

public class Temp 
{
	public Temp()
	{
		ArrayList<Integer> input = getInputList( 1000, 9999 ); 
		for( Integer i : input ) {
			if( blah( i ) == 7 ) {
				System.out.print( i + "," );
			}
		}
	}
	
	/*
	 * Generate input list of descending numbers 
	 */
	private static ArrayList<Integer> getInputList( final int min, final int max ) 
	{
		ArrayList<Integer> out = new ArrayList<Integer>(); 
		for( int i = min; i <= max; i++ ) {
			if( descending( i ) ) {
				out.add( i ); 
			}
		}
		return out; 
	}
	
	/*
	 * Generate blah. 
	 * The Blah of a number is the sum of the square of all its digits modulo 59. 
	 */
	private static int blah( final int i ) {
		final int sumOfSquares = sumOfSquares( i ); 
		final int blah = sumOfSquares % 59; 
		return blah; 
	}
	
	/*
	 * Generate digit-wise sum of squares. 
	 */
	private static int sumOfSquares( final int i ) {
		final int[] digits = getDigits( i ); 

		int sumOfSquares = 0;
		for( int pos = 0; pos < digits.length; pos++ ) {
			sumOfSquares += Math.pow( digits[ pos ], 2); 
		}
		return sumOfSquares; 
	}
	
	/*
	 * Test if integer is descending over digits: 
	 * A number is descending if its digits do not increase when read left to right. 
	 * For example, 5541 is descending but 5545 is not.
	 */
	private static boolean descending( final int i ) {
				
		boolean out = true;
		final int[] digits = getDigits( i ); 
		for( int pos = 1; pos < digits.length; pos++ ) { 
			if( digits[ pos - 1 ] < digits[ pos ] )
			{
				out = false; 
				break; 
			}
		}
		
		return out; 
	}
	
	/*
	 * Generate digit representation of integer. 
	 */
	private static int[] getDigits( final int i ) {
		final String value = String.valueOf( i );
		final int[] digits = new int[ value.length() ]; 
		
		for( int pos = 0; pos < value.length(); pos++ ) {	
			digits[ pos ] = Integer.parseInt( value.substring( pos, pos + 1 ) ); 
		}
		
		return digits; 
	}
	
	public static void main(String[] args) {
		new Temp();  
	}
}
