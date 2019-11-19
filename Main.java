/*
Name: Alexander Faucheux
CWID: 102-47-247
Date: October 16, 2019
Assignment #: 2
Description: A Neural Network designed to recognize handwritten digits.
 */


import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws IOException {
        Neural_Network NN = new Neural_Network();
        System.out.println("[1] Load last pre-trained network\n" +
                           "[2] Start training new network");

        Scanner input = new Scanner(System.in);
        String response = input.nextLine();

        // Loads last pre-trained network
        if(response.equals("1")) {
            // Initialize weights and biases
            NN.init_pretrained_variables();
        }


        else if(response.equals("2")) {
            // Initialize the weights and biases
            NN.init_random_variables();

            // Start training
            long starttime = System.currentTimeMillis();
            NN.run("train"); // Trains network
            long endtime = System.currentTimeMillis();
            System.out.println("\nTotal time elapsed: " + ((double) (endtime - starttime) / 1000));
        }

        String options1 = ("\nCurrent Network: Newly Trained\n" +
                           "[1] Load last saved pre-trained network\n" +
                           "[2] Start training new network\n" +
                           "[3] Run analysis on training data\n" +
                           "[4] Run analysis on testing data\n" +
                           "[5] Save current network (Note: Will overwrite last save)\n" +
                           "[6] exit\n");

        String options2 = ("\nCurrent Network: Pre-Trained\n" +
                           "[1] Start training new network\n" +
                           "[2] Run analysis on training data\n" +
                           "[3] Run analysis on testing data\n" +
                           "[4] Save current network (Note: Will overwrite last save)\n" +
                           "[5] exit\n");

        String currentOptions = response.equals("2") ? options1 : options2;

        while(true){
            int option;
            boolean isOptions1 = currentOptions.equals(options1);
            System.out.println(currentOptions);
            System.out.print("--> ");
            response = input.nextLine();
            try{
                option = Integer.parseInt(response);
                int maxOption = isOptions1 ? 6 : 5;
                if(option < 1 || option > maxOption){
                    System.out.println("Error: Invalid option. Please try again.\n\n");
                    continue;
                }
            }

            catch (Exception e){
                System.out.println("Error: Please enter a integer.\n\n");
                continue;
            }

            // Initialize pre-trained network
            if(isOptions1 && option == 1){
                NN.init_pretrained_variables();
                currentOptions = options2;
            }

            // Initialize and train Network
            else if((isOptions1 && option == 2) || (!isOptions1 && option == 1)){
                // Initialize the weights and biases
                NN.init_random_variables();

                // Start training
                long starttime = System.currentTimeMillis();
                NN.run("train"); // Trains network
                long endtime = System.currentTimeMillis();
                System.out.println("\nTotal time elapsed: " + ((double) (endtime - starttime) / 1000));
                currentOptions = options1;
            }

            // Run analysis on training data
            else if((isOptions1 && option == 3) || (!isOptions1 && option == 2)){
                NN.run("testTrain");
            }

            // Run analysis on testing data
            else if((isOptions1 && option == 4) || (!isOptions1 && option == 3)){
                NN.run("test");
            }

            // Save current Network State
            else if((isOptions1 && option == 5) || (!isOptions1 && option == 4)) {
                System.out.print("\nSaving... ");

                FileWriter weights_file1 = new FileWriter("weights1.csv");
                FileWriter weights_file2 = new FileWriter("weights2.csv");
                FileWriter bias_file1 = new FileWriter("bias1.csv");
                FileWriter bias_file2 = new FileWriter("bias2.csv");

                NN.save(1, 1, weights_file1);
                NN.save(1, 2, weights_file2);
                NN.save(2, 1, bias_file1);
                NN.save(2, 2, bias_file2);

                System.out.println("\rSaving... done\n");
            }

            // Exit the program
            else break;
        }
    }
}
