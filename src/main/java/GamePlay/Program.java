package GamePlay;

import java.util.HashSet;
import java.util.Scanner;

import org.bytedeco.librealsense.intrinsics;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import Game.GameStatus;
import Game.MinesweeperGame;
import Game.Pair;

import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.nativeblas.Nd4jCpu.static_bidirectional_rnn;

public class Program {
    static Scanner in;
    
    static int nChannels = 10;
    static int boardRows = 9;
	static int boardCols = 9;
	static int boardMines = 10;
	
    static int currentSample = 0;
    static int numSamples = 100;
    static int epochs = 2;
    static long playedGames = 0;
    static INDArray features, labels;
    static double[][][][] featuresArray;
    static double[][] labelsArray;
    
    static int gamesInSeries = 100;
    
    
    static MultiLayerNetwork model;

    public static void main(String[] args) {
        in = new Scanner(System.in);
        System.out.println("welcome to minesweepers");
        //playGameCLI(9, 9, 10);
        
        ConfigNetwork();
        train();

        //trainNetwork();
        /*FeedForwardNNLearner learner = new FeedForwardNNLearner();
        learner.trainNetwork();*/

        //in.close();
    }
    
    public static void ConfigNetwork() {
    	long seed = 123L;
    	
    	/*features = Nd4j.zeros(numSamples, nChannels, boardRows, boardCols); // TO FIX
    	labels = Nd4j.zeros(numSamples, boardRows * boardCols);*/
    	
    	featuresArray = new double[numSamples][nChannels][boardRows][boardCols];
    	labelsArray = new double[numSamples][boardRows * boardCols];
    	
    	//System.out.println(features.toString());
    	
    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                //.l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                //.updater(new Adam(1e-3))
                .list()
                
                .layer(new ConvolutionLayer.Builder(3, 3).padding(2,2)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        //Note that nIn need not be specified in later layers
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3).padding(2,2)
                        //Note that nIn need not be specified in later layers
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        //Note that nIn need not be specified in later layers
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new ConvolutionLayer.Builder(1, 1)
                        //Note that nIn need not be specified in later layers
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.SIGMOID)
                        .nOut(boardRows * boardCols).build())
                .layer(new OutputLayer.Builder()
                        .nOut(boardRows * boardCols)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .activation(Activation.SIGMOID)
                        .build())
                
                /*.layer(new DenseLayer.Builder().activation(Activation.SIGMOID)
                        .nOut(boardRows * boardCols).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(boardRows * boardCols)
                        .activation(Activation.SIGMOID)
                        .build())*/
                .setInputType(InputType.convolutional(boardCols,boardCols,nChannels))
                .build();
    	
    	model = new MultiLayerNetwork(conf);
        model.init();
    }
    
    public static void train() {
    	/*for(int i = 0; i < 10;i++) {
    		
    		model.fit(features, labels);
    		System.out.println("epoch " + (i + 1));
    	}*/
    	while(true) {
    		trainGame();
    	}
    }
    
    public static void trainGame() {
    	MinesweeperGame game = new MinesweeperGame(boardRows, boardCols, boardMines);
        game.loadGame(1, 2);
        GameStatus status;
        
        while((status = game.getGameStatus()) == GameStatus.PLAYING) {
        	
        	makeMove(game);
        }
        
        if(playedGames % gamesInSeries == 0) {
        	System.out.println("played games " + playedGames);
        }

        if(status != GameStatus.PLAYING) {
            playedGames++;
            
            /*if (status == GameStatus.LOST) {
            System.out.println("You LOST :(");
	        }
	        else */if (status == GameStatus.WON) {
	            System.out.println("You WON :) played: " + playedGames );
	            //in.nextLine();
	        }
        }
    }
    
    public static void makeMove(MinesweeperGame game) {
    	int[][] state = game.getGameState();
    	int[][] minesMatrix = game.getMineMatrix();
    	
    	/*System.out.println(game.toStringGameState());
    	System.out.println(game.toStringBoardValues());*/
    	
    	encodeState(state, featuresArray[currentSample]);
    	encodeOutputFlat(minesMatrix, labelsArray[currentSample]);
    	//labelsArray[currentSample][0] = copyAsDouble(minesMatrix);
    	double[][][][] inputArray = new double[1][][][];
    	inputArray[0] = featuresArray[currentSample];
    	incrementSampleSize();
    	
    	INDArray input = Nd4j.create(inputArray);
    	INDArray output = model.output(input);
    	
    	/*System.out.println("Output:\n\n" + output.toString());
    	in.nextLine();*/
    	
    	double[][] outputArray = decodeOutput(output.toDoubleMatrix()[0]);
    	//double[][] outputArray = output.get(NDArrayIndex.point(0)).get(NDArrayIndex.point(0)).toDoubleMatrix();
    	Pair bestMove = findMinProbabililtyBorder(outputArray, game.getBordersCellsArray());
    	game.unCover(bestMove.getRow(), bestMove.getCol());
    }
    
    public static void incrementSampleSize() {
    	currentSample++;
    	if(currentSample >= numSamples) {
    		features = Nd4j.create(featuresArray);
    		labels = Nd4j.create(labelsArray);
    		
    		for (int i = 0; i < epochs; i++) {
    			model.fit(features, labels);
			}
    		
    		currentSample = 0;
    	}
    }
    
    public static Pair findMinProbabililtyBorder(double[][] probabilityArray, Pair[] borderCells) {
    	double minProbability = 2.0;
    	Pair bestPosition = null;
    	for (Pair cell : borderCells) { 
    		if(probabilityArray[cell.getRow()][cell.getCol()] < minProbability) {
    			minProbability = probabilityArray[cell.getRow()][cell.getCol()];
    			bestPosition = cell;
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
    	
    	//print3Darray(inputArray);
    	
    }
    
    public static void setMatrixMatch(int[][] state, int matchVal, double[][] matchMatrix) {
    	for (int i = 0; i < boardRows; i++) {
    		for (int j = 0; j < boardCols; j++) {
    			matchMatrix[i][j] = state[i][j] == matchVal ? 1 : 0;
				
			}
		}
    }
    
    public static double[][] copyAsDouble(int[][] matrix) {
    	int rows = matrix.length, cols = matrix[0].length;
    	double[][] dst_matrix = new double[rows][cols];
    	
    	for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
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
            System.out.println(game.toStringBoardValues());
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
