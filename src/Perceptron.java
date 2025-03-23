import java.util.List;

/**
 * Classe Perceptron representa um modelo de perceptron que realiza operações básicas
 * de Machine learning, incluindo cálculos com funções de ativação como sigmoide.
 * Este perceptron utiliza pesos para ajustar a sua saída com base nas entradas fornecidas.
 */
public class Perceptron {
    private double[] weights;
    private double sigmoideValue; // Armazena o valor calculado pela função sigmoide.

    /**
     * Construtor que inicializa o perceptron com os pesos fornecidos.
     * O valor da sigmoide é inicialmente definido como 0.
     *
     * @param w Array de pesos para o perceptron.
     */
    public Perceptron(double[] w) {
        this.weights = w;
        this.sigmoideValue = 0.0;
    }

    /**
     * Retorna os pesos atuais do perceptron.
     *
     * @return Array de pesos.
     */
    public double[] getWeights() {
        return weights;
    }

    /**
     * Define novos valores para os pesos do perceptron.
     *
     * @param w Novo array de pesos.
     */
    public void setWeights(double[] w) {
        this.weights = w;
    }

    /**
     * Retorna o último valor calculado pela função sigmoide.
     *
     * @return Valor da sigmoide.
     */
    public double getSigmoideValue() {
        return sigmoideValue;
    }

    /**
     * Executa o cálculo de feedforward, combinando as entradas com os pesos
     * e aplicando a função de ativação sigmoide ao resultado.
     *
     * @param x Lista de valores de entrada.
     * @return Resultado da função sigmoide aplicada à soma ponderada.
     */
    public double feedForward(List<Double> x) {
        double sum = weights[0]; // Inicializa a soma com o peso de bias (primeiro peso).
        for (int i = 0; i < x.size(); i++) {
            sum += x.get(i) * weights[i + 1]; // Calcula a soma ponderada das entradas.
        }
        sigmoideValue = sigmoide(sum);
        return sigmoide(sum);
    }


    /**
     * Calcula o valor da função sigmoide para um dado z.
     * A sigmoide mapeia qualquer valor real para o intervalo (0, 1).
     *
     * @param z Valor de entrada.
     * @return Resultado da sigmoide.
     */
    public double sigmoide(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    /**
     * Calcula a derivada da função sigmoide para um dado valor de saída.
     * Utilizada em algoritmos de aprendizado para ajustar pesos.
     *
     * @param output Valor de saída da sigmoide.
     * @return Derivada da sigmoide.
     */
    public double sigmoideDerivative(double output) {
        return output * (1 - output);
    }

    /**
     * Aplica uma versão da função degrau usando como limiar o valor 0.5,
     * frequentemente associado à função sigmoide.
     *
     * @param sw Valor de saída da sigmoide.
     * @return 0 se o valor for menor que 0.5, caso contrário 1.
     */
    public int applyStepFunctionSigmoide(double sw) {
        int finalResult;
        if (sw < 0.5) {
            finalResult = 0;
        } else {
            finalResult = 1;
        }
        return finalResult;
    }
}