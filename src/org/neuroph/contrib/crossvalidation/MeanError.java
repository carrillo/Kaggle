package org.neuroph.contrib.crossvalidation;

import java.io.Serializable;

import org.neuroph.core.learning.error.ErrorFunction;

public class MeanError implements ErrorFunction, Serializable {

	private double totalSquaredErrorSum;
    private double n;

    public MeanError(double n) {
        this.n = n;
    }
    
    public void reset() {
        totalSquaredErrorSum = 0;
    }
        
    @Override
    public double getTotalError() {
        return Math.sqrt( totalSquaredErrorSum/n );
    }

    @Override
    public void addOutputError(double[] outputError) {
        double outputErrorSqrSum = 0;
        for (double error : outputError) {
            outputErrorSqrSum += Math.pow( error, 2 );
        }

        this.totalSquaredErrorSum += outputErrorSqrSum;
    }

}
