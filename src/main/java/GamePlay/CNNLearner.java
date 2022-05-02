package GamePlay;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.Scanner;

import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import Game.GameStatus;
import Game.MinesweeperGame;
import Game.Pair;

public class CNNLearner {
	int nChannels = 10;
    int boardRows = 9;
	int boardCols = 9;
	int boardMines = 10;
	
    int currentSample = 0;
    int numSamples = 200;
    long movesMade = 0;
    int epochs = 1;
    long playedGames = 0;
    INDArray features, labels;
    double[][][][] featuresArray;
    double[][][][] labelsArray;
    
    // for saving data
    int gamesInSeries = 1000;
    int gamesWonInSeries = 0;
    long movesInSeries = 0;
    long correctMovesInSeries = 0;
    long cellsToUnCoverInSeries = 0;
    String statsFileName = "stats.txt";
    String modelFileName = "model.txt";
    
    MultiLayerNetwork model;
    
    public CNNLearner(){
        
        featuresArray = new double[numSamples][nChannels][boardRows][boardCols];
    	labelsArray = new double[numSamples][1][boardRows][boardRows];
        
        //loadNetworkFromFile(); //use this line this to load an existing network
        //in.next();
        ConfigNetwork();	// use this to configure a new network
    }
    
    public void loadNetworkFromFile() {
    	try {
    		// save the model to be loaded in the model.txt file
			model = MultiLayerNetwork.load( new File("model.txt"), true);
			
			// transfer learning to a new network only used when necessary
			/*long seed = 1234;
			FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
					.seed(seed)
	                .weightInit(WeightInit.XAVIER)
	                .updater(new Adam())
	                .activation(Activation.RELU)
		            .build();
			//System.out.println(model.getnLayers());
			model=new TransferLearning.Builder(model)
					.fineTuneConfiguration(fineTuneConf)
		            .removeOutputLayer()
		            .addLayer(new CnnLossLayer.Builder(LossFunctions.LossFunction.MSE)
		            		.build())
		            .build();*/
			System.out.println(model.summary());
			//String var = model.
			//lastLayer.lossFunction(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR);
			
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.err.println("could not find model file");
		}
    }
    
    public void ConfigNetwork() {
    	long seed = 1234;
    	
    	//System.out.println(features.toString());
    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                //.l2(0.00001)
                .weightInit(WeightInit.XAVIER)
                //.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam())
                .activation(Activation.RELU)
                
               .list()
                //.layer(new ZeroPaddingLayer.Builder(5,5)
                //		.build())
                .layer(new ConvolutionLayer.Builder(3, 3).convolutionMode(ConvolutionMode.Same)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                		.stride(1,1)
                		.nIn(nChannels)
                        .nOut(25)
                        
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3).convolutionMode(ConvolutionMode.Same)
                        .nIn(30)
                        .activation(Activation.RELU)
                		.nOut(30)
                        .stride(1,1)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3).convolutionMode(ConvolutionMode.Same)
                        .nIn(30)
                        .activation(Activation.RELU)
                		.nOut(35)
                        .stride(1,1)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3).convolutionMode(ConvolutionMode.Same)
                        .nIn(35)
                        .activation(Activation.RELU)
                		.nOut(30)
                        .stride(1,1)
                        .build())
                .layer(new ConvolutionLayer.Builder(1, 1).convolutionMode(ConvolutionMode.Same)
                        //Note that nIn need not be specified in later layers
                		.stride(1,1)
                		.nIn(30)
                		.activation(Activation.RELU)
                		.nOut(64)
                        .build())
                /*.layer(new ConvolutionLayer.Builder(1, 1).convolutionMode(ConvolutionMode.Same)
                        //Note that nIn need not be specified in later layers
                		.stride(1,1)
                		.nIn(64)
                		.nOut(64)
                        .build())*/
                .layer(new ConvolutionLayer.Builder(1, 1)
                        //Note that nIn need not be specified in later layers
                		.stride(1,1)
                		.activation(Activation.RELU)
                		.nOut(1)
                        .build())
                
                .layer(new CnnLossLayer.Builder(LossFunctions.LossFunction.MSE)
                		//.activation(Activation.RELU)
                		.build())
                .setInputType(InputType.convolutional(boardRows,boardCols,nChannels))
                .backpropType(BackpropType.Standard)
                .build();
    	
    	model = new MultiLayerNetwork(conf);
        model.init();
        System.out.println(model.summary());
        System.out.println(model.getUpdater().getClass().getName());
        
    }
    
    public void train() {
    	while(true) {
    		trainGame();
    	}
    }
    
    public void trainGame() {
    	MinesweeperGame game = new MinesweeperGame(boardRows, boardCols, boardMines);
        game.loadGame(0, 0);
        GameStatus status;
        
        while((status = game.getGameStatus()) == GameStatus.PLAYING) {
        	
        	makeMove(game);
        	movesMade++;
        }
        
        if(playedGames % 100 == 0) {
        	System.out.println("played games " + playedGames + " " + LocalDateTime.now());
        	System.out.println("Moves Made " + movesMade);
        	movesMade = 0;
        	
        	//System.gc(); //add this for running tests
        }

        if(status != GameStatus.PLAYING) {
            playedGames++;
            cellsToUnCoverInSeries += game.getCellsToUncover();
            
            if(playedGames % gamesInSeries == 0) {
            	saveStatsToFile();
            	
            	saveModelToFile();  //Comment this for running tests
            }
            
            /*if (status == GameStatus.LOST) {
            System.out.println("You LOST :(");
	        }
	        else */if (status == GameStatus.WON) {
	        	gamesWonInSeries++;
	            System.out.println("----You WON :) played: " + playedGames );
	            //in.nextLine();
	        }
        }
    }
    
    public void makeMove(MinesweeperGame game) {
    	int[][] state = game.getGameState();
    	int[][] minesMatrix = game.getMineMatrix(); //comment this for running tests
    	
    	/*System.out.println(game.toStringGameState());
    	System.out.println(Arrays.deepToString(state));
    	System.out.println(game.toStringBoardValues());*/
    	
    	encodeState(state, featuresArray[currentSample]);
    	
    	labelsArray[currentSample][0] = copyAsDoubleWithPadding(minesMatrix); //comment this for running tests
    	double[][][][] inputArray = new double[1][][][];
    	inputArray[0] = featuresArray[currentSample];
    	incrementSampleSize();  //comment this for running tests
    	
    	INDArray input = Nd4j.create(inputArray);
    	INDArray output = model.output(input);
    	
    	//double[][] outputArray = decodeOutput(output.toDoubleMatrix()[0]);
    	double[][] outputArray = output.get(NDArrayIndex.point(0)).get(NDArrayIndex.point(0)).toDoubleMatrix();
    	//Pair bestMove = findMinProbabililtyBorder(outputArray, game.getBordersCellsArray());  // no guessing
    	Pair bestMove = findMinProbabililtyCell(outputArray, state); // allow guesses
    	
    	game.unCover(bestMove.getRow(), bestMove.getCol());
    	movesInSeries++;
    }
    
    public void incrementSampleSize() {
    	currentSample++;
    	if(currentSample >= numSamples) {
    		features = Nd4j.create(featuresArray);
    		labels = Nd4j.create(labelsArray);
    		
    		/*System.out.println("input: \n" + features.toString());
    		System.out.println("output: \n" + labels.toString());
    		in.nextLine();*/
    		
    		
    		for (int i = 0; i < epochs; i++) {
    			model.fit(features, labels);
			}
    		
    		currentSample = 0;
    		System.gc();
    	}
    }
    
    public Pair findMinProbabililtyCell(double[][] probabilityArray, int[][] state) {
    	double minProbability = 2.0;
    	Pair bestPosition = null;
    	for (int i = 0; i < probabilityArray.length; i++) {
    		for (int j = 0; j < probabilityArray[i].length; j++) {
    			if (probabilityArray[i][j] < minProbability && state[i][j] == MinesweeperGame.COVERED_CELL) {
    				minProbability = probabilityArray[i][j];
    				bestPosition = new Pair(i, j);
    			}
    		}
    	}
    	
    	if(bestPosition == null) {
    		System.err.println("Error: Probability > 1 calculated by neural net");
    		//in.nextLine();
    	}
    	
    	return bestPosition;
    }
    
    public Pair findMinProbabililtyBorder(double[][] probabilityArray, Pair[] borderCells) {
    	double minProbability = 2.0;
    	Pair bestPosition = null;
    	for (Pair cell : borderCells) { 
    		if(probabilityArray[cell.getRow()][cell.getCol()] < minProbability) {
    			minProbability = probabilityArray[cell.getRow()][cell.getCol()];
    			bestPosition = new Pair(cell.getRow(), cell.getCol());
    		}
    	}
    	if(bestPosition == null) {
    		System.err.println("Error: Probability > 1 calculated by neural net");
    		//in.nextLine();
    	}
    	return bestPosition;
    }
    
  //performs one hot encoding
    public void encodeState(int[][] state, double[][][] inputArray) {
    	int rows = state.length, cols = state[0].length;
    	//encode numbers from 0 to 8
    	// NOTE: careful when changing the index i
    	for (int i = 0; i < 9; i++) {
			setMatrixMatch(state, i, inputArray[i]);
		}
    	
    	setMatrixMatch(state, MinesweeperGame.COVERED_CELL, inputArray[9]);
    	
    	// Channel indicating the board size
    	/*for (int i = 0; i < boardRows; i++) {
    		for (int j = 0; j < boardCols; j++) {
    			inputArray[10][i][j] = 1;
				
			}
		}*/
    	
    }
    
    public void setMatrixMatch(int[][] state, int matchVal, double[][] matchMatrix) {
    	for (int i = 0; i < boardRows; i++) {
    		for (int j = 0; j < boardCols; j++) {
    			matchMatrix[i][j] = state[i][j] == matchVal ? 1 : 0;
				
			}
		}
    }
    
    public double[][] copyAsDoubleWithPadding(int[][] matrix) {
    	int rows = matrix.length, cols = matrix[0].length;
    	double[][] dst_matrix = new double[rows][cols];
    	
    	for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				dst_matrix[i][j] = matrix[i][j];
			}
		}
    	
    	return dst_matrix;
    }
    
    public double[][] removePadding(double[][] matrix) {
    	double[][] dst_matrix = new double[boardRows][boardCols];
    	for (int i = 0; i < boardRows; i++) {
			for (int j = 0; j < boardCols; j++) {
				dst_matrix[i][j] = matrix[i][j];
			}
		}
    	
    	return dst_matrix;
    }
    
    public void encodeOutputFlat(int[][] minesMatrix, double[] outputArray) {
    	int index = 0;
    	for (int i = 0; i < minesMatrix.length; i++) {
			for (int j = 0; j < minesMatrix[i].length; j++) {
				outputArray[index] = minesMatrix[i][j];
				index++;
			}
		}
    }
    
    public double[][] decodeOutput(double[] outputArray) {
    	double[][] outputMatrix = new double[boardRows][boardCols];
    	int row, col;
    	for (int i = 0; i < outputArray.length; i++) {
			row = i / boardCols;
			col = i % boardCols;
			outputMatrix[row][col] = outputArray[i];
		}
    	
    	return outputMatrix;
    }
    
    private void saveStatsToFile() {
    	try {
			BufferedWriter statsFile = new BufferedWriter(new FileWriter(statsFileName, true));
			String record = playedGames + ";" + gamesWonInSeries + ";" + movesInSeries + ";" + cellsToUnCoverInSeries + ";" + LocalDateTime.now() + ";";
			System.out.println(record);
			statsFile.write(record);
			statsFile.newLine();
			statsFile.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
    	
    	
    	// this series is complete
    	gamesWonInSeries = 0;
    	movesInSeries = 0;
        correctMovesInSeries = 0;
        cellsToUnCoverInSeries = 0;
    }
    
    private void saveModelToFile() {
    	//save model
    	try {
			model.save(new File(modelFileName), true);
		} catch (IOException e) {
			
			e.printStackTrace();
		}
    }
    
}
