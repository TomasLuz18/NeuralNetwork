import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static void main(String[] args)  {

        Scanner consoleScanner = new Scanner(System.in);
        String inputLine = consoleScanner.nextLine();
        double sigmoidValue = validation(inputLine);
        System.out.println(sigmoidValue);

        consoleScanner.close();

    }
    private static List<Double> parseInputLine(String line) {
        List<Double> list = new ArrayList<>();
        String[] parts = line.split(",");
        for (String p : parts) {
            double Pixelvalue = Double.parseDouble(p);
            if (Pixelvalue < 0){

                Pixelvalue = 0.0;
            }
            else if (Pixelvalue > 1){
                Pixelvalue = 1.0;
            }
            list.add(Pixelvalue);
        }
        return list;
    }
    private static double[] initializeWeights(int size) {
        double[] weights = new double[size];
        weights[0] = 0;
        for (int i = 1; i < size; i++) {
            weights[i] = (i)*1e-4;
        }
        return weights;
    }

    private static void trainAndTest(String filePathInputs, String filePathY) throws FileNotFoundException {
        Scanner scInputs = new Scanner(new File(filePathInputs));
        Scanner scExpectedY = new Scanner(new File(filePathY));
        Scanner scCount = new Scanner(new File(filePathY));
        int size = 400;
        int totalLines = 0;

        // Contar o total de linhas
        while (scCount.hasNextLine()) {
            scCount.nextLine();
            totalLines++;
        }
        scCount.close();

        int limit = (int) (totalLines * 0.9); // 90% treino, 10% teste

        double[] wH = initializeWeights(size + 1);
        double[] wO = initializeWeights(size + 2);

        List<Double> inputsTrain = new ArrayList<>();
        List<Integer> expectedYTrain = new ArrayList<>();
        List<List<Double>> inputsTest = new ArrayList<>();
        List<Integer> expectedYTest = new ArrayList<>();

        int count = 0;
        while (scInputs.hasNextLine()) {
            String lineInputs = scInputs.nextLine();
            String lineExpectedOutput = scExpectedY.nextLine();

            List<Double> parsedLine = parseInputLine(lineInputs);

            if (count < limit) {
                // Dados de treino
                expectedYTrain.add(Integer.parseInt(lineExpectedOutput));
                // Adicionar todos os valores parseados a inputsTrain
                inputsTrain.addAll(parsedLine);
            } else {
                // Dados de teste
                expectedYTest.add(Integer.parseInt(lineExpectedOutput));
                inputsTest.add(parsedLine);
            }
            count++;
        }

        scInputs.close();
        scExpectedY.close();

        Perceptron h = new Perceptron(wH);
        Perceptron o = new Perceptron(wO);
        NeuralNetwork neuralNetwork = new NeuralNetwork(h, o);

        // Treinar
        neuralNetwork.train(inputsTrain, expectedYTrain);

        // Testar no conjunto de teste
        int result = 0;
        for (int i = 0; i < inputsTest.size(); i++) {
            double predictedY = neuralNetwork.testNetwork(inputsTest.get(i));
            int y = expectedYTest.get(i);
            if ((int) predictedY == y) {
                result++;
            }
        }
        double accuracy = (((double) result / expectedYTest.size()) * 100);
        System.out.println("Precisão do treino: " + accuracy + "%");
    }


    public static double validation(String inputLine) {

        List<Double> inputValues = parseInputLine(inputLine);


        double[] wH = initializeWeights(401);


        Perceptron h = new Perceptron(wH);

        h.feedForward(inputValues);
        double sigmoide = h.getSigmoideValue();


        return Math.round(sigmoide * 100.0) / 100.0;
    }







}