package GamePlay;

import java.util.HashSet;
import java.util.Scanner;

import Game.GameStatus;
import Game.MinesweeperGame;

public class Program {
    static Scanner in;

    public static void main(String[] args) {
        in = new Scanner(System.in);
        System.out.println("welcome to minesweepers");
        //playGameCLI(9, 9, 10);
        
        CNNLearner learner = new CNNLearner();
        learner.train();

        //trainNetwork();
        /*FeedForwardNNLearner learner = new FeedForwardNNLearner();
        learner.trainNetwork();*/

        //in.close();
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
