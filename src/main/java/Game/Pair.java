package Game;

public class Pair {
    int row;
    int col;
    public Pair(int row,int col){
        this.row = row;
        this.col = col;
    }



    public int getRow() {
        return row;
    }



    public void setRow(int row) {
        this.row = row;
    }



    public int getCol() {
        return col;
    }



    public void setCol(int b) {
        this.col = b;
    }



    public boolean equals(Object o) {
        if (o instanceof Pair) {
            Pair p = (Pair)o;
            return p.row == row && p.col == col;
        }
        return false;
    }

    public int hashCode() {
        return row * 31 + col;
    }

    public String toString() {
        return "(" + row + ", " + col + ")";
    }
}