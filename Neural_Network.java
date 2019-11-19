import java.lang.Math;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;




class Neural_Network {
    // Variables that define the parameters of the neural network
    // Note: Matrix_Operations is from Matrix.java, a separate file I created
    final Matrix_Operations matrixOP = new Matrix_Operations();
    final String trainFile = "mnist_train.csv";
    final String testFile = "mnist_test.csv";
    final int batch_size = 10;
    final int input_layer_size = 784;
    final int hidden_layer_size = 30 ;
    final int output_layer_size = 10;
    final int train_input_size = 50000;
    final int test_input_size = 10000;
    final int trainEpochMax = 30;
    final double learning_rate = 3.0;

    final double[] bias1 = new double[hidden_layer_size];
    final double[] bias2 = new double[output_layer_size];
    final double[][] weights1 = new double[hidden_layer_size][input_layer_size];
    final double[][] weights2 = new double[output_layer_size][hidden_layer_size];

    // Class variables to be changed/initialized later
    int x;
    int epochMax;
    int[] totalNums;
    String[] lines;

    // Initialize variables to use pre-trained network
    void init_pretrained_variables() throws FileNotFoundException {
        // Read stored values
        Scanner w1 = new Scanner(new File("weights1.csv"));
        Scanner w2 = new Scanner(new File("weights2.csv"));
        Scanner b1 = new Scanner(new File("bias1.csv"));
        Scanner b2 = new Scanner(new File("bias2.csv"));

        String[] w1Lines = new String[hidden_layer_size];
        String[] w2Lines = new String[output_layer_size];
        String b1Line;
        String b2Line;

        // Read weight files, create array of lines for each file
        int index = 0;
        while (w1.hasNextLine()) {
            w1Lines[index] = w1.nextLine();
            index++;
        }

        index = 0;
        while (w2.hasNextLine()) {
            w2Lines[index] = w2.nextLine();
            index++;
        }

        // Bias files are only 1 line.
        b1Line = b1.nextLine();
        b2Line = b2.nextLine();

        w1.close();
        w2.close();
        b1.close();
        b2.close();

        // Fill weight matrices with values from files
        for (int i = 0; i < weights1.length; i++) {
            String[] parsedLine = w1Lines[i].split(",");
            for (int k = 0; k < weights1[0].length; k++)
                weights1[i][k] = Double.parseDouble(parsedLine[k]);
        }

        for (int i = 0; i < weights2.length; i++) {
            String[] parsedLine = w2Lines[i].split(",");
            for (int k = 0; k < weights2[0].length; k++)
                weights2[i][k] = Double.parseDouble(parsedLine[k]);
        }

        // Fill bias arrays with values from files
        String[] parsedLine = b1Line.split(",");
        String[] parsedLine2 = b2Line.split(",");

        for (int i = 0; i < bias1.length; i++) {
            bias1[i] = Double.parseDouble(parsedLine[i]);
        }

        for (int i = 0; i < bias2.length; i++) {
            bias2[i] = Double.parseDouble(parsedLine2[i]);
        }

    }

    // Initialize variables to train new network
    void init_random_variables() {
        Random random = new Random();

        // Initialize weights
        for (int i = 0; i < weights1.length; i++) {
            for (int k = 0; k < weights1[0].length; k++)
                weights1[i][k] = random.nextDouble() * 2 - 1;
        }

        for (int i = 0; i < weights2.length; i++) {
            for (int k = 0; k < weights2[0].length; k++)
                weights2[i][k] = random.nextDouble() * 2 - 1;
        }

        // Initialize biases
        for (int i = 0; i < bias1.length; i++)
            bias1[i] = random.nextDouble() * 2 - 1;

        for (int i = 0; i < bias2.length; i++)
            bias2[i] = random.nextDouble() * 2 - 1;
    }

    // Shuffles list of inputs from file
    void shuffle_list(double[][] inputs, double[][] outputs) {
        // Will only shuffle list if program is using training set
        if (inputs.length > test_input_size) {
            List<String> newList = Arrays.asList(lines);
            Collections.shuffle(newList);
            lines = newList.toArray(new String[0]);
        }

        // Parses each line from file and adds to inputs matrix and outputs matrix
        // Inputs matrix: each row represent a different input and consist of 784 greyscale values divided by 255.
        // Outputs matrix: each row represent a different expected output proportionate to the the rows in the input matrix
        //                 the expected value is derived from the first element from line array
        //                 the expected value is turned into a label: Each index is a digit 0 or 1.
        //                 If element at index = 1, digit = index

        for (int i = 0; i < lines.length; i++) {
            String[] line = lines[i].split(",");
            for (int k = 1; k < line.length; k++) {
                double num = Double.parseDouble(line[k]);
                inputs[i][k - 1] = num / 255;
            }
            outputs[i] = label_converter(Integer.parseInt(line[0]));
            if (x == 0) totalNums[Integer.parseInt(line[0])] += 1;
        }

        if (x == 0) x++;
    }

    // Initialize input and output matrices
    void init_in_out(String file, double[][] inputs, double[][] outputs) throws FileNotFoundException {
        System.out.print("\nInitializing...");
        lines = new String[inputs.length];
        Scanner f = new Scanner(new File(file));
        int index = 0;
        while (f.hasNextLine() && index < inputs.length) {
            lines[index] = f.nextLine();
            index++;
        }
        f.close();

        // Used to shuffle the list and populate the two matrices
        shuffle_list(inputs, outputs);

        System.out.println("\rInitializing... Done.");
        System.out.println();
    }

    // Activation Function
    double sigmordal(double num) {
        return 1 / (1 + Math.pow(Math.E, -num));
    }

    // Get cost of layer output
    double getCost(double[] expected_output, double[][] actual_output) {
        double cost = 0;
        for (int i = 0; i < expected_output.length; i++)
            cost += Math.pow(expected_output[i] - actual_output[i][0], 2);
        cost *= 0.5;
        return cost;
    }

    // Checks to see if output is correct and increments appropriate counter if so.
    void check_accuracy(double[] exp_out, double[] act_out, int[] correctNums) {
        for (int i = 0; i < exp_out.length; i++) {
            if (exp_out[i] == 1.0 && act_out[i] == Arrays.stream(act_out).max().getAsDouble()) {
                correctNums[i] += 1;
                break;
            }
        }
    }

    // If input is an array.
    double[][] getLayerOutput(double[] input, double[][] weights, double[] biases) {
        return getLayerOutput(matrixOP.transpose(input), weights, biases);
    }

    // Takes inputs, weights, and biases and returns layer output.
    double[][] getLayerOutput(double[][] input, double[][] weights, double[] biases) {
        double[][] z = matrixOP.productOf(weights, input, biases);
        for (double[] arr : z) {
            arr[0] = sigmordal(arr[0]);
        }
        return z;
    }

    // Returns weight gradient based on inputs and appropriate bias gradient
    double[][] getWeightGradient(double[] input, double[] bias_gradient) {
        double[][] gradient = new double[bias_gradient.length][input.length];
        for (int i = 0; i < bias_gradient.length; i++) {
            for (int k = 0; k < input.length; k++) {
                double product = input[k] * bias_gradient[i];
                gradient[i][k] = product;
            }
        }
        return gradient;
    }

    // Returns bias gradient for output layer
    double[] get_Output_BiasGradient(double[] exp_output, double[][] act_output) {
        double[] gradient = new double[act_output.length];
        for (int i = 0; i < act_output.length; i++)
            gradient[i] = (act_output[i][0] - exp_output[i]) * act_output[i][0] * (1 - act_output[i][0]);

        return gradient;
    }

    // Returns bias gradient for hidden layer
    double[] get_Hidden_BiasGradient(double[] output_grad, double[][] act_output, double[][] weights) {
        double[] gradient = new double[act_output.length];
        for (int i = 0; i < act_output.length; i++) {
            double total = 0;
            for (int k = 0; k < weights.length; k++) {
                total += weights[k][i] * output_grad[k];
            }
            gradient[i] = total * (act_output[i][0] * (1 - act_output[i][0]));
        }
        return gradient;
    }

    // Fills a mini batch matrix with specified range of inputs.
    void fill_miniBatch(double[][] mini_batch, double[][] filler, int range1, int range2) {
        int k = 0;
        for (int i = range1; i < range2; i++, k++) {
            mini_batch[k] = filler[i].clone();
        }
    }

    // Revises biases based on gradients
    void revise_biases(double[] biases, ArrayList<double[]> biasGradients) {
        for (int i = 0; i < biases.length; i++) {
            double oldBias = biases[i];
            double sum_of_gradients = 0;
            for (int k = 0; k < biasGradients.size(); k++) {
                sum_of_gradients += biasGradients.get(k)[i];
            }
            biases[i] = oldBias - (learning_rate / batch_size) * sum_of_gradients;
        }
    }

    // Revise weights based on gradients
    void revise_weights(double[][] weights, ArrayList<double[][]> weightGradients) {
        for (int row = 0; row < weights.length; row++) {
            for (int column = 0; column < weights[0].length; column++) {
                double oldWeight = weights[row][column];
                double sum_of_gradients = 0;
                for (int k = 0; k < weightGradients.size(); k++) {
                    sum_of_gradients += weightGradients.get(k)[row][column];
                }
                weights[row][column] = oldWeight - (learning_rate / batch_size) * sum_of_gradients;
            }
        }
    }

    // Converts a digit to an array used as an output label
    double[] label_converter(int num) {
        double[] base = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        base[num] = 1;
        return base;
    }

    // Saves current state of neural network into appropriate weight and bias csv files
    // type 1 == weight, type 2 == bias
    void save(int type, int layer, FileWriter saveFile) throws IOException {
        if (type == 1 && layer < 3) {
            double[][] weights = (layer == 1 ? weights1 : weights2);

            for (int i = 0; i < weights.length; i++) {
                for (int k = 0; k < weights[0].length; k++)
                    saveFile.write(weights[i][k] + ",");

                saveFile.write("\n");
            }

            saveFile.close();

        } else if (type == 2 && layer < 3) {
            double[] bias = (layer == 1 ? bias1 : bias2);
            
            for (int i = 0; i < bias.length; i++)
                saveFile.write(bias[i] + ",");

            saveFile.close();
        }
    }

    /*
    Runs a test or trains the network
    Because this is a hybrid of testing and training,
    there are measures that prevent "test mode" to run functions specifically for training.
    When it is used for testing, epochMax is set to 1, and prevents these functions from being called:
    -- Any function that deals with gradients, the shuffle function, and the revise functions.
     */
    void run(String type) throws FileNotFoundException {
        x = 0;
        totalNums = new int[10];
        double[][] inputs = new double[type.equals("train") || type.equals("testTrain")? train_input_size : test_input_size][input_layer_size];
        double[][] answers = new double[type.equals("train") || type.equals("testTrain")? train_input_size : test_input_size][output_layer_size];

        if(type.equals("train")) {
            epochMax = trainEpochMax;
            init_in_out(trainFile, inputs, answers);
            System.out.print("Training... Completed 0/" + epochMax);
        }

        else if(type.equals("testTrain")){
            epochMax = 1;
            init_in_out(trainFile, inputs, answers);
            System.out.print("Testing... ");
        }

        else if(type.equals("test")){
            epochMax = 1;
            init_in_out(testFile, inputs, answers);
            System.out.print("Testing... ");
        }

        // Training begins
        // Loop for epoch.
        for (int epoch = 0; epoch < epochMax; epoch++) {
            long startTime = System.currentTimeMillis();
            int[] correctNums = new int[10];
            // Loop for each mini_batch.  Mini-batch size of 10 for this test
            for (int inputIndex = 0; inputIndex < inputs.length; inputIndex += batch_size) {
                // Initialize lists to store gradients for each input at each layer
                // *Will only be used if epochMax > 1*
                ArrayList<double[]> hidden_biasGradients = new ArrayList<>();
                ArrayList<double[]> output_biasGradients = new ArrayList<>();
                ArrayList<double[][]> hidden_weightGradients = new ArrayList<>();
                ArrayList<double[][]> output_weightGradients = new ArrayList<>();

                // initialize mini batch matrices
                double[][] mini_batch_input = new double[batch_size][inputs[0].length]; // mini batch of inputs
                double[][] mini_batch_output = new double[batch_size][answers[0].length]; // mini batch of outputs

                // fill mini batches
                fill_miniBatch(mini_batch_input, inputs, inputIndex, inputIndex + batch_size);
                fill_miniBatch(mini_batch_output, answers, inputIndex, inputIndex + batch_size);

                // Loop for each row in a mini_batch matrix
                for (int batchIndex = 0; batchIndex < mini_batch_input.length; batchIndex++) {
                    // Initialize input and expected output arrays
                    double[] input = mini_batch_input[batchIndex];
                    double[] expect_out = mini_batch_output[batchIndex];

                    // Get outputs for each layer
                    double[][] result = getLayerOutput(input, weights1, bias1);
                    double[][] result2 = getLayerOutput(result, weights2, bias2);

                    // Check accuracy of output.  If output is correct, increment correctNums
                    check_accuracy(expect_out, matrixOP.toArray(result2), correctNums);

                    // Runs only during training
                    if(epochMax > 1) {
                        // Get bias gradients for each batch
                        double[] layer2_biasGradient = get_Output_BiasGradient(expect_out, result2);
                        double[] layer1_biasGradient = get_Hidden_BiasGradient(layer2_biasGradient, result, weights2);

                        // Get weight gradients for each batch
                        double[][] layer2_weightGradient = getWeightGradient(matrixOP.toArray(result), layer2_biasGradient);
                        double[][] layer1_weightGradient = getWeightGradient(input, layer1_biasGradient);

                        // Store gradients
                        hidden_biasGradients.add(layer1_biasGradient);
                        hidden_weightGradients.add(layer1_weightGradient);
                        output_biasGradients.add(layer2_biasGradient);
                        output_weightGradients.add(layer2_weightGradient);
                    }
                }

                // Runs only during training
                if(epochMax > 1) {
                    // Revise weights and biases
                    revise_biases(bias1, hidden_biasGradients);
                    revise_biases(bias2, output_biasGradients);
                    revise_weights(weights1, hidden_weightGradients);
                    revise_weights(weights2, output_weightGradients);
                }
            }

            int output_sum = Arrays.stream(correctNums).sum();
            int total_sum = Arrays.stream(totalNums).sum();
            long endTime = System.currentTimeMillis();
            double totalTime = endTime - startTime;

            System.out.print(String.format("\repoch %d:\n0 = %d/%d\t1 = %d/%d\t2 = %d/%d\t3 = %d/%d\t4 = %d/%d\t5 = %d/%d\n6 = %d/%d\t7 = %d/%d\t8 = %d/%d\t9 = %d/%d\nTime elapsed: %.4f sec\nAccuracy = %d/%d = %.4f",
                    epoch + 1, correctNums[0], totalNums[0], correctNums[1], totalNums[1], correctNums[2], totalNums[2], correctNums[3], totalNums[3], correctNums[4], totalNums[4],
                    correctNums[5], totalNums[5], correctNums[6], totalNums[6], correctNums[7], totalNums[7], correctNums[8], totalNums[8], correctNums[9], totalNums[9], (totalTime / 1000),
                    output_sum, total_sum, ((double) output_sum / (double) total_sum) * 100));

            System.out.println("%\n");

            // Runs only during training
            if(epochMax > 1) {
                System.out.print("\rTraining... Completed " + (epoch + 1) + "/" + epochMax);
                shuffle_list(inputs, answers);
            }

            else
                System.out.println("\rTesting...  done.");
        }
    }
}