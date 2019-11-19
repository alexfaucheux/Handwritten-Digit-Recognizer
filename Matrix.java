/*
Designed to do matrix operations on matrices containing double type values.
Used to make the main code shorter and easier to read.
 */

class Matrix_Operations {

    // Takes two matrices and returns their product
    double[][] productOf(double[][] matrix1, double[][] matrix2, double[]bias) {
        double[][] product = new double[matrix1.length][matrix2[0].length];
        for (int i = 0; i < matrix1.length; i++) {
            for (int j = 0; j < matrix2[0].length; j++) {
                for (int k = 0; k < matrix1[0].length; k++) {
                    product[i][j] += matrix1[i][k] * matrix2[k][j];
                }
                product[i][j] += (bias != null ? bias[i] : 0);
            }
        }
        return product;
    }

    // Prints matrix in a row x column fashion
    void printMatrix(double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int k = 0; k < matrix[0].length; k++) {
                System.out.print(matrix[i][k] + "\t");
            }
            System.out.println();
        }
    }

    // Copies a matrix
    double[][] copyMatrix(double[][] matrix){
        double[][] newMatrix = new double[matrix.length][matrix[0].length];
        for(int i = 0; i < matrix.length; i++){
            newMatrix[i] = matrix[i].clone();
        }
        return newMatrix;
    }

    // Transpose an array by making a matrix with 1 column
    // Each row contains one element from the array
    double[][] transpose(double[] array){
        double[][] newMatrix = new double[array.length][1];
        for(int i = 0; i < array.length; i++){
            newMatrix[i][0] = array[i];
        }
        return newMatrix;
    }

    // Returns a transposed version of a general matrix
    double[][] transpose(double[][] matrix){
        double[][] newMatrix = new double[matrix[0].length][matrix.length];
        for(int i = 0; i < matrix[0].length; i++){
            for(int k = 0; k < matrix.length; k++) {
                newMatrix[i][k] = matrix[k][i];
            }
        }
        return newMatrix;
    }

    // Takes a single column matrix and outputs an array
    double[] toArray(double[][] matrix){
        if(matrix[0].length == 1) {
            double[] newArray = new double[matrix.length];
            for(int i = 0; i < matrix.length; i++){
                newArray[i] = matrix[i][0];
            }
            return newArray;
        }

        else
            return null;
    }

}