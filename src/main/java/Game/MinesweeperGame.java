package Game;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.Scanner;

import org.bytedeco.librealsense.intrinsics;


public class MinesweeperGame {
    /**
     * Game configuration values:
     * 0 - 8: indicate that there are n adjacent cells containing mines. For instance, 1 means that
     * 		  exactly 1 of the adjacent cells contains a mine. 0 means that no adjacent cells have a mine.
     * -1: 	  means that the cell is a mine
     * -2:	  only set on displayBoard, means that the cell is to be revealed.
     */

    public static final short MINE = -1;
    public static final short COVERED_CELL = -2;
    public static final short OUT_OF_RANGE_CELL = 0;


    int numRows;
    int numCols;
    int numMines;

    int[][] boardValues;
    int[][] gameState;
    int[][] uncoveredNeighboursCount;

    GameStatus gameStatus;
    int cellsToUncover;
    HashSet<Pair> borderCells;
    
    
    Scanner inScanner;

    public MinesweeperGame(int numRows, int numCols, int numMines){
        if (numMines > numRows * numCols / 2) {
            numMines = numRows * numCols / 2;
        }

        this.numRows = numRows;
        this.numCols = numCols;
        this.numMines = numMines;
        this.cellsToUncover = numCols * numRows - numMines;
        this.gameStatus = GameStatus.PLAYING;

        this.gameState = new int[this.numRows][this.numCols];
        this.uncoveredNeighboursCount = new int[this.numRows][this.numCols];

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                gameState[i][j] = COVERED_CELL;
                uncoveredNeighboursCount[i][j] = 0;
            }
        }

        borderCells = new HashSet<Pair>();
        inScanner = new Scanner(System.in);
    }

    public int[][] getGameState() {
        return cloneIntMatrix(gameState, numRows, numCols);
    }

    public int[][] getUncoveredNeighboursCount() {
        return cloneIntMatrix(uncoveredNeighboursCount, numRows, numCols);
    }

    private int[][] cloneIntMatrix(int[][] src, int numrows, int numcols) {
        int[][] newMatrix = new int[numrows][numcols];
        for(int i=0; i<src.length; i++)
            for(int j=0; j<src[i].length; j++)
                newMatrix[i][j]=src[i][j];

        return newMatrix;
    }

    public HashSet<Pair> getBorderCells() {
        return borderCells;
    }
    
    public Pair[] getBordersCellsArray() {
    	ArrayList<Pair> borders = new ArrayList<Pair>();
    	for (int i = 0; i < numRows; i++) {
			for (int j = 0; j < numCols; j++) {
				if(isBorder(i, j)) {
					borders.add(new Pair(i, j));
				}
			}
		}
    	
    	return borders.toArray(new Pair[borders.size()]);
    }
    
    // method to remove later
    public boolean isRightMove(int row, int col) {
    	if (row < 0 || row >= numRows || col < 0 || col >= numCols) {
        	return false;
        }   
        else if (gameState[row][col] != COVERED_CELL) {
        	return false;
        }
    	
    	return true;
    }

    public GameStatus getGameStatus() {
        return gameStatus;
    }

    public int getCellsToUncover() {
        return cellsToUncover;
    }

    public void loadGame(int startRow, int startCol) {

        boardValues = new int[this.numRows][this.numCols];
        gameState = new int[this.numRows][this.numCols];

        gameStatus = GameStatus.PLAYING;
        cellsToUncover = numCols * numRows - numMines;

        for (int i = 0; i < numMines; i++) {
            placeSingleMine(startRow, startCol);
        }

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                gameState[i][j] = COVERED_CELL;
            }
        }

        unCover(startRow, startCol);

    }

    private void placeSingleMine(int startX, int startY) {
        boolean minePlaced = false;
        Random random = new Random();

        while (!minePlaced) {
            int i = random.nextInt(numRows);
            int j = random.nextInt(numCols);

            // the random position must not be starting position (starting position is always safe) and
            // must not already contain a mine
            if (boardValues[i][j] != MINE && !(i == startX && j == startY) && !isAdjacent(i, j, startX, startY)) {
                boardValues[i][j] = MINE;
                minePlaced = true;

                for (int k = -1; k <= 1; k++) {
                    for (int l = -1; l <= 1; l++) {
                        if (!(k == 0 && l == 0))
                            incrementMineCount(i + k, j + l);
                    }
                }
            }
        }
    }

    private boolean isAdjacent(int x1, int y1, int x2, int y2) {
        for (int k = -1; k <= 1; k++) {
            for (int l = -1; l <= 1; l++) {
                if(x1 + k == x2 && y1 + l == y2) {
                    return true;
                }
            }
        }

        return false;
    }

    private void incrementMineCount(int i, int j) {
        if(i < 0 || i >= numRows || j < 0 || j >= numCols || boardValues[i][j] == MINE) {
            return;
        }

        boardValues[i][j] += 1;
    }

    /**
     * Only use this for training.
     * @param row
     * @param col
     * @return
     */
    public boolean isMine(int row, int col) {
        if (row < 0 || row >= numRows || col < 0 || col >= numCols) {
            return false;
        }

        return boardValues[row][col] == MINE;
    }
    
    /**
     * Only use this for training.
     * @return
     */
    public int[][] getMineMatrix() {
    	int[][] mines = new int[this.numRows][this.numCols];
    	for (int i = 0; i < boardValues.length; i++) {
			for (int j = 0; j < boardValues[i].length; j++) {
				mines[i][j] = boardValues[i][j] == MINE ? 1 : 0;
			}
		}
    	
    	return mines;
    }

    public GameStatus unCover(int row, int col) {
        if (gameStatus != GameStatus.PLAYING)
            return gameStatus;
        else if (row < 0 || row >= numRows || col < 0 || col >= numCols) {
        	//System.out.println("Error: ("+ row  + ", " + col + ") move out of the box");
        	return gameStatus;
        }   
        else if (gameState[row][col] != COVERED_CELL) {
        	/*System.out.println("GameState: \n" + toStringGameState());
        	System.out.println("Error: ("+ row  + ", " + col + ") box already uncovered");
        	System.out.println(borderCells.toString());
        	inScanner.next();*/
        	return gameStatus;
        }
        else if (boardValues[row][col] == 0) {
            unCoverCell(row, col);
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {

                    if (!(k == 0 && l == 0))
                        unCover(row + k, col + l);
                }
            }
        }
        else {
            unCoverCell(row, col);
        }

        return gameStatus;
    }

    private void unCoverCell(int row, int col) {
        gameState[row][col] = boardValues[row][col];
        incrementUncoveredNeighbourCount(row, col);
        if (boardValues[row][col] != MINE) {
            cellsToUncover--;
            gameStatus = cellsToUncover == 0 ? GameStatus.WON : gameStatus;
        }
        else {
            gameStatus = GameStatus.LOST;
        }
    }

    private void incrementUncoveredNeighbourCount(int row, int col) {
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int rowi = row + i, colj = col + j;
                if (isValidCell(rowi, colj )) {
                    if(rowi != row || colj != col) {
                        uncoveredNeighboursCount[rowi][colj] += 1;
                    }
                    checkBorder(rowi, colj);
                }

            }
        }
    }

    private void checkBorder(int row, int col) {
        Pair pair = new Pair(row, col);

        if(gameState[row][col] != COVERED_CELL) {
            borderCells.remove(pair);
            return;
        }

        // corners have at 3 neighbouring cells
        if(row % (numRows - 1) == 0 && col % (numCols -1) == 0) {
            if(uncoveredNeighboursCount[row][col] >= 3) { //strictly speaking == 3 but let's leave >= for safety
                borderCells.remove(pair);
                return;
            }
        }

        // top, bottom rows and left, right columns have 5 neighbour cells
        if(row % (numRows - 1) == 0 || col % (numCols -1) == 0) {
            if(uncoveredNeighboursCount[row][col] >= 5) {  //strictly speaking == 5 but let's leave >= for safety
                borderCells.remove(pair);
                return;
            }

        }

        if(uncoveredNeighboursCount[row][col] >= 8) {
            borderCells.remove(pair);
        }
        else if(uncoveredNeighboursCount[row][col] > 0) {
            borderCells.add(pair);
        }
    }
    
    private boolean isBorder(int row, int col) {
    	if(gameState[row][col] != COVERED_CELL) {
    		return false;
    	}
    	
    	for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int rowi = row + i, colj = col + j;
                if (isValidCell(rowi, colj )) {
                    if((rowi != row || colj != col) && gameState[rowi][colj] != COVERED_CELL) {
                        return true;
                    }
                }

            }
        }
    	
    	return false;
    }

    private boolean isValidCell(int row, int col) {
        if (row < 0 || row >= numRows || col < 0 || col >= numCols) {
            return false;
        }
        return true;
    }

    public String toStringBoardValues() {
        return matrixToString(boardValues);
    }

    public String toStringGameState() {
        return matrixToString(gameState);
    }

    private String matrixToString(int[][] matrix) {
        String str = " ".repeat(6);
        //String threeSpaces = " ".repeat(3);
        String twoSpaces= " ".repeat(2);
        String rowTopString = " ".repeat(6) + ("|" + " ".repeat(5)).repeat(matrix[0].length) + "|";
        String rowBottomString = " ".repeat(6) + ("|" + "_".repeat(5)).repeat(matrix[0].length) + "|";

        for (int i = 0; i < matrix[0].length; i++) {
            str += twoSpaces + String.format("%2d", i) + twoSpaces;
        }

        str += "\n" + " ".repeat(6) + "_".repeat(matrix[0].length * 6) + "\n";

        for (int i = 0; i < matrix.length; i++) {
            str += rowTopString + "\n";
            str += twoSpaces + String.format("%2d", i) + twoSpaces;

            for (int j = 0; j < matrix[i].length; j++) {
                String valString = "  ";
                switch (matrix[i][j]) {
                    case COVERED_CELL:
                        valString = " ?";
                        break;
                    case MINE:
                        valString = " M";
                        break;
                    case 0:
                        valString = twoSpaces;
                        break;
                    default:
                        valString = String.format("%2d", matrix[i][j]);
                }
                str += "|" + " " + valString + twoSpaces;
            }
            str += "|\n" + rowBottomString + "\n";
        }

        return str;
    }

}
