import java.util.Random;

// Clase que representa una Neurona
class Neurona {
    private double[] pesos; // Pesos de la neurona
    private double bias;    // Bias de la neurona
    private double salida;  // Salida de la neurona

    public Neurona(int numEntradas) {
        // Inicializamos los pesos y el bias con valores aleatorios
        Random random = new Random();
        pesos = new double[numEntradas];
        for (int i = 0; i < pesos.length; i++) {
            pesos[i] = random.nextDouble() * 2 - 1; // Valores entre -1 y 1
        }
        bias = random.nextDouble() * 2 - 1; // Bias inicial aleatorio
    }

    public double calcularSalida(double[] entradas) {
        // Calcula la salida de la neurona aplicando la función de activación
        double suma = bias;
        for (int i = 0; i < entradas.length; i++) {
            suma += entradas[i] * pesos[i];
        }
        salida = funcionSigmoide(suma);
        return salida;
    }

    public double funcionSigmoide(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public double derivadaSigmoide(double x) {
        return x * (1 - x);
    }

    public double[] getPesos() {
        return pesos;
    }

    public void actualizarPesos(double[] entradas, double tasaAprendizaje, double delta) {
        // Actualizamos los pesos y el bias con el delta
        for (int i = 0; i < pesos.length; i++) {
            pesos[i] += tasaAprendizaje * delta * entradas[i];
        }
        bias += tasaAprendizaje * delta;
    }

    public double getSalida() {
        return salida;
    }
}

// Clase que representa una Capa de la red neuronal
class Capa {
    private Neurona[] neuronas;

    public Capa(int numNeuronas, int numEntradasPorNeurona) {
        neuronas = new Neurona[numNeuronas];
        for (int i = 0; i < numNeuronas; i++) {
            neuronas[i] = new Neurona(numEntradasPorNeurona);
        }
    }

    public double[] calcularSalidas(double[] entradas) {
        double[] salidas = new double[neuronas.length];
        for (int i = 0; i < neuronas.length; i++) {
            salidas[i] = neuronas[i].calcularSalida(entradas);
        }
        return salidas;
    }

    public Neurona[] getNeuronas() {
        return neuronas;
    }
}

// Clase principal que implementa la Red Neuronal
class RedNeuronal {
    private Capa[] capas;
    private double tasaAprendizaje;

    public RedNeuronal(int[] estructura, double tasaAprendizaje) {
        this.tasaAprendizaje = tasaAprendizaje;
        capas = new Capa[estructura.length - 1];
        for (int i = 0; i < capas.length; i++) {
            capas[i] = new Capa(estructura[i + 1], estructura[i]);
        }
    }

    public double[] predecir(double[] entradas) {
        double[] salidas = entradas;
        for (Capa capa : capas) {
            salidas = capa.calcularSalidas(salidas);
        }
        return salidas;
    }

    public void entrenar(double[][] datosEntrenamiento, double[][] etiquetas, int epocas) {
        for (int epoca = 0; epoca < epocas; epoca++) {
            for (int i = 0; i < datosEntrenamiento.length; i++) {
                // Paso hacia adelante (forward pass)
                double[] salidas = predecir(datosEntrenamiento[i]);

                // Paso hacia atrás (backpropagation)
                double[] errores = new double[etiquetas[i].length];
                for (int j = 0; j < errores.length; j++) {
                    errores[j] = etiquetas[i][j] - salidas[j];
                }

                // Retropropagación del error
                for (int capaIndex = capas.length - 1; capaIndex >= 0; capaIndex--) {
                    Capa capaActual = capas[capaIndex];
                    double[] nuevosErrores = new double[capaActual.getNeuronas()[0].getPesos().length];

                    for (int n = 0; n < capaActual.getNeuronas().length; n++) {
                        Neurona neurona = capaActual.getNeuronas()[n];
                        double delta = errores[n] * neurona.derivadaSigmoide(neurona.getSalida());

                        // Actualizar pesos
                        double[] entradas = (capaIndex == 0) ? datosEntrenamiento[i] : capas[capaIndex - 1].calcularSalidas(datosEntrenamiento[i]);
                        neurona.actualizarPesos(entradas, tasaAprendizaje, delta);

                        // Calcular errores para la capa anterior
                        for (int k = 0; k < nuevosErrores.length; k++) {
                            nuevosErrores[k] += delta * neurona.getPesos()[k];
                        }
                    }
                    errores = nuevosErrores;
                }
            }
        }
    }
}

// Clase Main para probar la red neuronal
public class Main {
    public static void main(String[] args) {
        // Estructura de la red: 2 entradas, 2 neuronas ocultas, 1 salida
        int[] estructura = {2, 2, 1};
        RedNeuronal red = new RedNeuronal(estructura, 0.5);

        // Datos de entrenamiento (compuerta lógica AND)
        double[][] datosEntrenamiento = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        double[][] etiquetas = {
            {0},
            {0},
            {0},
            {1}
        };

        // Entrenar la red
        red.entrenar(datosEntrenamiento, etiquetas, 10000);

        // Probar la red
        System.out.println("Resultados:");
        for (double[] datos : datosEntrenamiento) {
            double[] resultado = red.predecir(datos);
            System.out.printf("Entrada: %.0f, %.0f -> Salida: %.5f%n", datos[0], datos[1], resultado[0]);
        }
    }
}