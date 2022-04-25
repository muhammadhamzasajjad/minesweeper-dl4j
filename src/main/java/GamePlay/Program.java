package GamePlay;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.Scanner;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import Game.GameStatus;
import Game.MinesweeperGame;
import Game.Pair;

import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Program {
    static Scanner in;
    
    static int nChannels = 10;
    static int boardRows = 9;
	static int boardCols = 9;
	static int boardMines = 10;
	
    static int currentSample = 0;
    static int numSamples = 200;
    static long movesMade = 0;
    static int epochs = 1;
    static long playedGames = 0;
    static INDArray features, labels;
    static double[][][][] featuresArray;
    static double[][][][] labelsArray;
    
    //saving data
    static int gamesInSeries = 1000;
    static int gamesWonInSeries = 0;
    static long movesInSeries = 0;
    static long correctMovesInSeries = 0;
    static long cellsToUnCoverInSeries = 0;
    static String statsFileName = "stats.txt";
    static String modelFileName = "model.txt";
    
    static MultiLayerNetwork model;

    public static void main(String[] args) {
        in = new Scanner(System.in);
        System.out.println("welcome to minesweepers");
        //playGameCLI(9, 9, 10);
        
        featuresArray = new double[numSamples][nChannels][boardRows][boardCols];
    	labelsArray = new double[numSamples][1][boardRows][boardRows];
        
        loadNetworkFromFile();
        //in.next();
        //ConfigNetwork();
        train();

        //trainNetwork();
        /*FeedForwardNNLearner learner = new FeedForwardNNLearner();
        learner.trainNetwork();*/

        //in.close();
    }
    
    public static void loadNetworkFromFile() {
    	try {
			model = MultiLayerNetwork.load( new File("model.txt"), true);
			
			// transfer learning to a new network with 
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
    
    public static void ConfigNetwork() {
    	long seed = 1234;
    	
    	/*features = Nd4j.zeros(numSamples, nChannels, boardRows, boardCols); // TO FIX
    	labels = Nd4j.zeros(numSamples, boardRows * boardCols);*/
    	
    	//System.out.println(features.toString());
    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                //.l2(0.00001)
                .weightInit(WeightInit.XAVIER)
                //.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.updater(new Sgd(0.001))
                .updater(new Adam())
                //.updater(new )
                //.
                //.l2(0.00001)
                //.updater(Updater.ADAM)
                .activation(Activation.RELU)
                
                .list()
                //.layer(new ZeroPaddingLayer.Builder(5,5)
                //		.build())
                .layer(new ConvolutionLayer.Builder(3, 3).convolutionMode(ConvolutionMode.Same)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                		.stride(1,1)
                		.nIn(nChannels)
                        .nOut(15)
                        
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3).convolutionMode(ConvolutionMode.Same)
                        .nIn(15)
                        .activation(Activation.RELU)
                		.nOut(20)
                        .stride(1,1)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3).convolutionMode(ConvolutionMode.Same)
                        .nIn(20)
                        .activation(Activation.RELU)
                		.nOut(10)
                        .stride(1,1)
                        .build())
                /*.layer(new ConvolutionLayer.Builder(3, 3).convolutionMode(ConvolutionMode.Same)
                        .nIn(50)
                        .activation(Activation.RELU)
                		.nOut(50)
                        .stride(1,1)
                        .build())
                .layer(new ConvolutionLayer.Builder(1, 1).convolutionMode(ConvolutionMode.Same)
                        //Note that nIn need not be specified in later layers
                		.stride(1,1)
                		.nIn(50)
                		.activation(Activation.RELU)
                		.nOut(50)
                        .build())*/
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
                		//.activation(Activation.SIGMOID)
                		.build())
                /*.layer(new DenseLayer.Builder().activation(Activation.SIGMOID)
                        .nOut(boardRows * boardCols).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(boardRows * boardCols)
                        .activation(Activation.SIGMOID)
                        .build())*/
                .setInputType(InputType.convolutional(boardRows,boardCols,nChannels))
                .backpropType(BackpropType.Standard)
                .build();
    	
    	model = new MultiLayerNetwork(conf);
        model.init();
        System.out.println(model.summary());
        System.out.println(model.getUpdater().getClass().getName());
        
    }
    
    public static void train() {
    	while(true) {
    		trainGame();
    	}
    }
    
    public static void trainGame() {
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
        	
        	System.gc(); //add this for running tests
        }

        if(status != GameStatus.PLAYING) {
            playedGames++;
            cellsToUnCoverInSeries += game.getCellsToUncover();
            
            if(playedGames % gamesInSeries == 0) {
            	saveStatsToFile();
            	
            	//saveModelToFile();  //Comment this for running tests
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
    
    public static void makeMove(MinesweeperGame game) {
    	int[][] state = game.getGameState();
    	int[][] minesMatrix = game.getMineMatrix(); //comment this for running tests
    	
    	/*System.out.println(game.toStringGameState());
    	System.out.println(Arrays.deepToString(state));
    	System.out.println(game.toStringBoardValues());*/
    	
    	encodeState(state, featuresArray[currentSample]);
    	
    	//labelsArray[currentSample][0] = copyAsDoubleWithPadding(minesMatrix); //comment this for running tests
    	double[][][][] inputArray = new double[1][][][];
    	inputArray[0] = featuresArray[currentSample];
    	//incrementSampleSize();  //comment this for running tests
    	
    	INDArray input = Nd4j.create(inputArray);
    	INDArray output = model.output(input);
    	/*System.out.println("Output:\n\n" + output.toString());
    	in.nextLine();*/
    	
    	//double[][] outputArray = decodeOutput(output.toDoubleMatrix()[0]);
    	double[][] outputArray = output.get(NDArrayIndex.point(0)).get(NDArrayIndex.point(0)).toDoubleMatrix();
    	//Pair bestMove = findMinProbabililtyBorder(outputArray, game.getBordersCellsArray());
    	Pair bestMove = findMinProbabililtyCell(outputArray, state); // allow guesses
    	
    	/*System.out.println("Best move: " + bestMove.getRow() + ", " + bestMove.getCol());
    	in.nextLine();*/
    	
    	game.unCover(bestMove.getRow(), bestMove.getCol());
    	movesInSeries++;
    }
    
    public static void incrementSampleSize() {
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
    
    public static Pair findMinProbabililtyCell(double[][] probabilityArray, int[][] state) {
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
    		in.nextLine();
    	}
    	
    	return bestPosition;
    }
    
    public static Pair findMinProbabililtyBorder(double[][] probabilityArray, Pair[] borderCells) {
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
    		in.nextLine();
    	}
    	return bestPosition;
    }
    
    //performs one hot encoding
    public static void encodeState(int[][] state, double[][][] inputArray) {
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
    	
    	//print3Darray(inputArray);
    	
    }
    
    public static void setMatrixMatch(int[][] state, int matchVal, double[][] matchMatrix) {
    	for (int i = 0; i < boardRows; i++) {
    		for (int j = 0; j < boardCols; j++) {
    			matchMatrix[i][j] = state[i][j] == matchVal ? 1 : 0;
				
			}
		}
    }
    
    public static double[][] copyAsDoubleWithPadding(int[][] matrix) {
    	int rows = matrix.length, cols = matrix[0].length;
    	double[][] dst_matrix = new double[rows][cols];
    	
    	for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				dst_matrix[i][j] = matrix[i][j];
			}
		}
    	
    	return dst_matrix;
    }
    
    public static double[][] removePadding(double[][] matrix) {
    	double[][] dst_matrix = new double[boardRows][boardCols];
    	for (int i = 0; i < boardRows; i++) {
			for (int j = 0; j < boardCols; j++) {
				dst_matrix[i][j] = matrix[i][j];
			}
		}
    	
    	return dst_matrix;
    }
    
    public static void encodeOutputFlat(int[][] minesMatrix, double[] outputArray) {
    	int index = 0;
    	for (int i = 0; i < minesMatrix.length; i++) {
			for (int j = 0; j < minesMatrix[i].length; j++) {
				outputArray[index] = minesMatrix[i][j];
				index++;
			}
		}
    }
    
    public static double[][] decodeOutput(double[] outputArray) {
    	double[][] outputMatrix = new double[boardRows][boardCols];
    	int row, col;
    	for (int i = 0; i < outputArray.length; i++) {
			row = i / boardCols;
			col = i % boardCols;
			outputMatrix[row][col] = outputArray[i];
		}
    	
    	return outputMatrix;
    }
    
    public static void print3Darray(double[][][] arr) {
    	String comma;
    	System.out.print("[");
    	for (int i = 0; i < arr.length; i++) {
    		System.out.print("[");
			for (int j = 0; j < arr[i].length; j++) {
				System.out.print("[");
				for (int j2 = 0; j2 < arr[i][j].length; j2++) {
					comma = j2 < arr[i][j].length - 1 ? ", " : "";
					System.out.print(arr[i][j][j2] + comma);
				}
				comma = j < arr[i].length - 1 ? "," : "";
				System.out.print("]" + comma+ "\n");
			}
			comma = i < arr.length - 1 ? "," : "";
			System.out.print("]" + comma+ "\n");
		}
    	System.out.println("]");
    }
    
    
    /*public static double[][] decodeFlatOutputToMineMatrix(INDArray flatOutput) {
    	flatOutput.size(0);
    	for (int i = 0; i < array.length; i++) {
			
		}
    }*/
    
    
    private static void saveStatsToFile() {
    	try {
			BufferedWriter statsFile = new BufferedWriter(new FileWriter(statsFileName, true));
			String record = playedGames + ";" + gamesWonInSeries + ";" + movesInSeries + ";" + cellsToUnCoverInSeries + ";" + LocalDateTime.now() + ";";
			System.out.println(record);
			statsFile.write(record);
			statsFile.newLine();
			statsFile.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	
    	
    	// this series is complete
    	gamesWonInSeries = 0;
    	movesInSeries = 0;
        correctMovesInSeries = 0;
        cellsToUnCoverInSeries = 0;
    }
    
    private static void saveModelToFile() {
    	//save model
    	try {
			model.save(new File(modelFileName), true);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
    
    
    

    public static void playGameCLI(int numRows, int numCols, int numMines) {
        int row, col;
        int[] rowcol;
        HashSet<String> set = new HashSet<String>();
        boolean gameLoaded = false;
        MinesweeperGame game = new MinesweeperGame(numRows, numCols, numMines);

        while (!gameLoaded) {
            System.out.println(game.toStringGameState());
            rowcol = readMoveCLI();
            if(rowcol[0] != -1 && rowcol[1] != -1) {
                game.loadGame(rowcol[0], rowcol[1]);
                gameLoaded = true;
            }

        }

        GameStatus status;
        while((status = game.getGameStatus()) == GameStatus.PLAYING) {
            //System.out.println(game.toStringBoardValues());
            System.out.println(game.toStringGameState());
            //System.out.print("Empty boxes left: " + game.getBoxesToUncover());

            rowcol = readMoveCLI();
            //System.out.println("input encoding: "+ inputs.toString());
            game.unCover(rowcol[0], rowcol[1]);

        }
        System.out.println(game.toStringGameState());
        if (status == GameStatus.LOST) {
            System.out.print("You LOST :(");
        }
        else if (status == GameStatus.WON) {
            System.out.print("You WON :)");
        }
    }

    public static int[] readMoveCLI() {
        int[] rowcol = new int[2];
        System.out.print("Your move(r c): ");
        String move = in.nextLine();
        String[] rc = move.split(" ");
        if(rc.length != 2) {
            System.err.println("invalid move");
            rowcol[0] = rowcol[1] = -1;
        }
        try {
            rowcol[0] = Integer.parseInt(rc[0]);
            rowcol[1] = Integer.parseInt(rc[1]);
        } catch (Exception e) {
            System.err.println("invalid move");
            rowcol[0] =rowcol[1] = -1;
        }

        return rowcol;
    }

}
