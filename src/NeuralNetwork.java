import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A classe NeuralNetwork representa uma rede neural simples com uma camada oculta
 * e uma camada de saída, cada uma implementada como um perceptron.
 * A rede utiliza uma função de ativação sigmoide e suporta o treino usando o método de backpropagation.
 */
public class NeuralNetwork {
    private Perceptron hidden;
    private Perceptron output;
    private static final double LEARNINGRATE = 0.05;
    private final String CSV_FILE = "mse.csv"; // Arquivo para guardar o erro médio quadrático durante o treino.

    /**
     * Construtor para inicializar a network com perceptrons para a camada oculta e de saída.
     *
     * @param hidden Perceptron da camada oculta.
     * @param output Perceptron da camada de saída.
     */
    public NeuralNetwork(Perceptron hidden, Perceptron output) {
        this.hidden = hidden;
        this.output = output;
    }

    /**
     * Realiza o processo de feedforward na network.
     *
     * @param inputs Lista de entradas para a rede.
     * @return Saída da camada de saída após o processamento.
     */
    public double feedForward(List<Double> inputs) {
        double hiddenOutput = this.hidden.feedForward(inputs); // Calcula a saída da camada oculta.
        List<Double> outputInputs = new ArrayList<>(inputs); // Copia as entradas para a camada de saída.
        outputInputs.add(hiddenOutput); // Adiciona a saída da camada oculta como entrada adicional.
        return this.output.feedForward(outputInputs); // Calcula e retorna a saída da rede.
    }

    /**
     * Aplica uma função de step function ao resultado fornecido.
     *
     * @param result Valor para aplicar a função degrau.
     * @return 0 se o resultado for menor que 0, caso contrário 1.
     */
    public int applyStepFunction(double result){
        int finalresult = 0;
        if (result >= 0.5) {
            finalresult = 1;
        }
        return finalresult;
    }

    /**
     * Treina a network usando o método de retropropagação.
     *
     * @param inputs Lista de entradas do conjunto de treino.
     * @param y Lista de saídas esperadas do conjunto de treino.
     */
    public void train(List<Double> inputs, List<Integer> y) {
        try (FileWriter writer = new FileWriter(CSV_FILE)) {
            writer.append("Iteration,MSE\n"); // Cabeçalho do arquivo do MSE.
            int iteration = 0;
            int size = 400; // Tamanho do input.
            int n = y.size(); // Número total de exemplos do dataset.
            double threshold = 1e-6; // Erro mínimo aceitável.
            double mse = Double.MAX_VALUE;
            int maxIterations = 100000;

            System.out.println("A treinar... Demora algum tempo");
            while (mse > threshold && iteration < maxIterations) {
                mse = 0;

                for (int i = 0; i < n; i++) {
                    List<Double> inputsIteration = inputs.subList(i * size, (i + 1) * size); // Divide as entradas por lote.

                    double predictY = feedForward(inputsIteration); // Faz a previsão da saída.

                    double error = predictY - y.get(i);
                    mse += 0.5 * Math.pow(error, 2); // Atualiza o MSE.

                    double deltaOutput = error * output.sigmoideDerivative(predictY); // Calcula o gradiente para a camada de saída.

                    double deltaHidden = deltaOutput * output.getWeights()[output.getWeights().length - 1] * hidden.sigmoideDerivative(hidden.getSigmoideValue()); // Calcula o gradiente para a camada oculta.

                    double[] weightsOutput = output.getWeights(); //
                    weightsOutput[0] -= LEARNINGRATE * deltaOutput; // Atualiza o bias da camada de saída.
                    for (int j = 0; j < inputsIteration.size() - 1; j++) { // Atualiza os pesos associados às entradas.
                        weightsOutput[j + 1] -= LEARNINGRATE * deltaOutput * inputsIteration.get(j);
                    }
                    weightsOutput[weightsOutput.length - 1] -= LEARNINGRATE * deltaOutput * hidden.getSigmoideValue(); // Atualiza o peso associado à saída da camada oculta.
                    output.setWeights(weightsOutput); // Define os novos pesos da camada de saída.

                    double[] weightsHidden = hidden.getWeights();
                    weightsHidden[0] -= LEARNINGRATE * deltaHidden; // Atualiza o bias da camada oculta.
                    for (int j = 0; j < inputsIteration.size(); j++) { // Atualiza os pesos associados às entradas.
                        weightsHidden[j + 1] -= LEARNINGRATE * deltaHidden * inputsIteration.get(j);
                    }
                    hidden.setWeights(weightsHidden); // Define os novos pesos da camada oculta.
                }

                mse /= n; // Calcula o MSE.
                writer.append(String.valueOf(iteration)) // Escreve a iteração e o erro no ficheiro CSV.
                        .append(",")
                        .append(String.valueOf(mse))
                        .append("\n");
                iteration++;
            }
            System.out.println("Sessão de treino terminada!");
            saveWeightsToFile("hidden_weights.csv", hidden.getWeights());
            saveWeightsToFile("output_weights.csv", output.getWeights());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Guarda os pesos fornecidos em um arquivo.
     *
     * @param filename Nome do arquivo onde os pesos serão salvos.
     * @param weights Array de pesos a serem salvos.
     */
    private void saveWeightsToFile(String filename, double[] weights) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(filename))) {
            for (int i = 0; i < weights.length; i++) {
                bw.write(String.valueOf(weights[i]));
                if (i < weights.length - 1) bw.write(",");             }
            bw.newLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Testa a network com os valores de entrada fornecidos.
     *
     * @param inputs Lista de entradas para teste.
     * @return 1 se a saída for maior ou igual a 0.5, caso contrário 0.
     */
    public int testNetwork(List<Double> inputs) {
        double predictedY = feedForward(inputs);
        return applyStepFunction(predictedY);
    }

    /**
     * Retorna uma representação textual da network, incluindo os pesos finais
     * dos perceptrons oculto e de saída.
     *
     * @return String com os pesos formatados.
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("\n\nPesos finais do perceptron HIDDEN:\n");
        appendWeights(sb, hidden);
        sb.append("\nPesos finais do perceptron OUTPUT:\n");
        appendWeights(sb, output);
        return sb.toString();
    }

    /**
     * Método auxiliar para formatar os pesos de um perceptron.
     *
     * @param sb StringBuilder para acumular o texto formatado.
     * @param perceptron Perceptron cujos pesos serão formatados.
     */
    private void appendWeights(StringBuilder sb, Perceptron perceptron) {
        double[] weights = perceptron.getWeights();
        sb.append(String.format("Bias: %.4f\n", weights[0]));
        for (int i = 1; i < weights.length; i++) {
            sb.append(String.format("Peso %d: %.4f\n", i, weights[i]));
        }
    }
}
