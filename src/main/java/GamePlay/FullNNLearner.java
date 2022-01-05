package GamePlay;

import java.util.Scanner;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class FullNNLearner {
	int numCells;
	int boardRows;
	int boardCols;
	int numMines;
    int currentSample;
    int numSamples;
    long playedGames;
    INDArray features, labels;
    MultiLayerNetwork model;
    
    //saving data
    int gamesInSeries;
    int gamesWonInSeries;
    long movesInSeries;
    long correctMovesInSeries;
    long cellsToUnCoverInSeries;
    String statsFileName;
    String modelFileName;
    
    Scanner input;
    
	public FullNNLearner() {
		currentSample = 0;
        numSamples = 128;
        playedGames = 0;
        boardRows = 9;
        boardCols = 9;
        numMines = 10;
        numCells = boardRows * boardCols;
        features = Nd4j.zeros(numSamples, numCells);
        labels = Nd4j.zeros(numSamples, 1);

        //configNetwork();
        
        gamesInSeries = 5000;
        gamesWonInSeries = 0;
        movesInSeries = 0;
        correctMovesInSeries = 0;
        cellsToUnCoverInSeries = 0;
        statsFileName = "stats.txt";
        modelFileName = "model.txt";
        
        input = new Scanner(System.in);
	}
}
