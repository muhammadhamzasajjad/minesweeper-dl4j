package GamePlay;

import Game.GameStatus;
import Game.MinesweeperGame;
import Game.Pair;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.Scanner;

public class FeedForwardNNLearner {
    final int numInputs = 24;
    int currentSample;
    int movesMade;
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

    public FeedForwardNNLearner() {
        currentSample = 0;
        numSamples = 128;
        movesMade = 0;
        playedGames = 0;
        features = Nd4j.zeros(numSamples, numInputs);
        labels = Nd4j.zeros(numSamples, 1);

        configNetwork();
        //loadNetworkFromFile();
        
        gamesInSeries = 1000;
        gamesWonInSeries = 0;
        movesInSeries = 0;
        correctMovesInSeries = 0;
        cellsToUnCoverInSeries = 0;
        statsFileName = "stats.txt";
        modelFileName = "model.txt";
        
        input = new Scanner(System.in);
    }
    
    public void loadNetworkFromFile() {
    	try {
			model = MultiLayerNetwork.load( new File("model.txt"), true);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.err.println("could not find model file");
		}
    }

    public void configNetwork() {
        int seed = 12345;
        int nEpochs = 5;
        double learningRate = 0.1;
        int numOutputs = 1;

        //
        // Configure the model with number layers and nodes in each layer.
        //
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(learningRate))
                .l2(0.0001)
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(70)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(70).nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(100).nOut(90)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(90).nOut(45)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        .nIn(45).nOut(numOutputs).build())
                .build();  // 24/70/100/90/45/1
        
        //run the model
        model = new MultiLayerNetwork(conf);
        model.init();
        System.out.println(model.summary());

    }

    public void trainNetwork() {
    	
        while(true) {
            this.trainGame();
        }

    }

    private void trainGame() {
    	
    	MinesweeperGame game = new MinesweeperGame(9, 9, 10);
        game.loadGame(1, 2);
        GameStatus status;
        
        while((status = game.getGameStatus()) == GameStatus.PLAYING) {
        	
            makeMove(game);
            movesMade++;
        }

        if(status != GameStatus.PLAYING) {
            playedGames++;
            cellsToUnCoverInSeries += game.getCellsToUncover();
            
            if(playedGames % 100 == 0) {
            	System.gc();
                System.out.println("played games: " + playedGames + " moves Made: " + movesMade);
                movesMade = 0;
            }
            
            if(playedGames % gamesInSeries == 0) {
            	saveStatsToFile();
            	
            	saveModelToFile();
            }
            
            /*if (status == GameStatus.LOST) {
            System.out.println("You LOST :(");
	        }
	        else */if (status == GameStatus.WON) {
	        	gamesWonInSeries++;
	            System.out.println("You WON :) played: " + playedGames );
	            System.out.println("Moves: " + movesInSeries + ";" + correctMovesInSeries);
	            //in.nextLine();
	        }
        }        
        
        
    }

    private void makeMove(MinesweeperGame game) {
        int[][] gameState = game.getGameState();
        Pair[] borderCells = game.getBordersCellsArray();
        double lowestmineProbability = 2.0; // we will always find a lower probability
        Pair bestMove = null;
        INDArray bestInputs = null;
        /*System.out.println("border: " + borderCells.toString());
        System.out.println("game state: \n"  + game.toStringGameState());
        input.nextLine();*/


        for (Pair cell : borderCells) {
            double[][] inputVals = encodeInput(gameState, cell.getRow(), cell.getCol());

            INDArray inputs = Nd4j.create(inputVals);
            INDArray output = model.output(inputs);
            double mineProbability = output.getDouble(0,0);

            //if(playedGames < 1000) {
                double correctOutput = game.isMine(cell.getRow(), cell.getCol()) ? 1.0 : 0.0;
                if (Math.abs(correctOutput - mineProbability) >= 0.5) {
                    INDArray trainOutput = Nd4j.zeros(1, 1);
                    trainOutput.putScalar(0, 0, correctOutput);
                    addTrainingRow(inputs.getRow(0, false), trainOutput.getRow(0, false));
                }
            //}


            if(mineProbability < lowestmineProbability) {
                lowestmineProbability = mineProbability;
                bestMove = cell;
                bestInputs = inputs;
            }
        }

        if(bestMove != null) {
            //System.out.println("best prob: " + lowestmineProbability);
        	if(!game.isRightMove(bestMove.getRow(), bestMove.getCol())) {
        		System.out.println("Error Wrong bestMove: " + bestMove.toString());
        		System.out.println(game.toStringGameState());
        		input.nextLine();
        	}
            game.unCover(bestMove.getRow(), bestMove.getCol());
            movesInSeries++;

            GameStatus status = game.getGameStatus();
            INDArray trainOutput = Nd4j.zeros(1, 1);
            if (status == GameStatus.LOST) {
                trainOutput.putScalar(0, 0, 1.0);
                
            } else {
                trainOutput.putScalar(0, 0, 0.0);
                correctMovesInSeries++;
            }

            addTrainingRow(bestInputs.getRow(0, false), trainOutput.getRow(0, false));
        }

    }

    private void addTrainingRow(INDArray input, INDArray output) {
        features.putRow(currentSample, input);
        labels.putRow(currentSample, output);
        currentSample++;
        if(currentSample >= numSamples) {
            currentSample = 0;
            for (int i = 0; i < 4; i++) {
                model.fit(features, labels);
            }
            
        }
    }

    private double[][] encodeInput(int[][] gameState, int row, int col) {
        double[][] values = new double[1][numInputs];
        int vali = 0;
        for(int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                int rowi = row + i, colj = col + j;
                if(rowi != row || colj != col) {
                    if(isValidCell(gameState, rowi, colj)) {
                        values[0][vali] = ( gameState[rowi][colj] - ( - 2.0 ) ) / (8.0 - (-2.0));
                        //values[0][vali] = ( gameState[rowi][colj] );
                    }
                    else {
                        values[0][vali] = ( MinesweeperGame.OUT_OF_RANGE_CELL - ( - 2.0 ) ) / (8.0 - (-2.0));
                        //values[0][vali] = ( MinesweeperGame.OUT_OF_RANGE_CELL );
                    }
                    vali++;
                }
            }
        }

        return values;
    }

    private boolean isValidCell(int[][] matrix,  int row, int col) {
        int numRows = matrix.length, numCols = matrix[0].length;
        if (row < 0 || row >= numRows || col < 0 || col >= numCols) {
            return false;
        }
        return true;
    }
    
    private void saveStatsToFile() {
    	try {
			BufferedWriter statsFile = new BufferedWriter(new FileWriter(statsFileName, true));
			String record = playedGames + ";" + gamesWonInSeries + ";" + movesInSeries + ";" + correctMovesInSeries + ";" + cellsToUnCoverInSeries + ";" + LocalDateTime.now() + ";";
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
    
    private void saveModelToFile() {
    	//save model
    	try {
			model.save(new File(modelFileName), true);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }

}
