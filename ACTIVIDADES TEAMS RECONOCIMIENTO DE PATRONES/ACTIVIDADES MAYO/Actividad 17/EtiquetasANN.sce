// Red Neuronal Feedforward usando Toolbox ANN

// --- Cargar el Toolbox ANN ---
atomsLoad("ANN_Toolbox");

// --- Definir la estructura de la red neuronal ---
// Configura el número de neuronas: entrada, oculta, salida.
// 13 características de entrada y 5 clases de salida (0, 1, 2, 3, 4).
n_entradas = 13;  // Número de neuronas en la capa de entrada (características)
n_ocultas = 10;   // Número de neuronas en la capa oculta (ejemplo)
n_salidas = 5;    // Número de neuronas en la capa de salida (una por cada clase 0-4)

// Vector que representa la estructura para las funciones del toolbox.
N = [n_entradas, n_ocultas, n_salidas];

// --- Datos de Ejemplo (20 Muestras Hardcodeadas) ---
// Un conjunto de 20 muestras seleccionadas del dataset reprocessed.hungarian.data.
// Cada fila es una muestra con 13 características. Valores -9 reemplazados por 0.
X_demo = [
// Muestras con etiqueta 0
40, 1, 2, 140, 289, 0, 0, 172, 0, 0, 0, 0, 0;
37, 1, 2, 130, 283, 0, 1, 98, 0, 0, 0, 0, 0;
54, 1, 3, 150, 0, 0, 0, 122, 0, 0, 0, 0, 0;
39, 1, 3, 120, 339, 0, 0, 170, 0, 0, 0, 0, 0;
45, 0, 2, 130, 237, 0, 0, 170, 0, 0, 0, 0, 0;

// Muestras con etiqueta 1
49, 0, 3, 160, 180, 0, 0, 156, 0, 1, 2, 0, 0;
37, 1, 4, 140, 207, 0, 0, 130, 1, 1.5, 2, 0, 0;
49, 1, 4, 140, 234, 0, 0, 140, 1, 1, 2, 0, 0;
38, 1, 4, 110, 196, 0, 0, 166, 0, 0, 0, 0, 0;
36, 1, 2, 120, 267, 0, 0, 160, 0, 3, 2, 0, 0;

// Muestras con etiqueta 2
58, 1, 2, 136, 164, 0, 1, 99, 1, 2, 2, 0, 0;
52, 1, 4, 120, 182, 0, 0, 150, 0, 0, 0, 0, 0;
52, 1, 4, 160, 329, 0, 0, 92, 1, 1.5, 2, 0, 0;
48, 1, 4, 160, 355, 0, 0, 99, 1, 2, 2, 0, 0;
62, 1, 2, 140, 271, 0, 0, 152, 0, 1, 1, 0, 0;

// Muestras con etiqueta 3
48, 0, 4, 138, 214, 0, 0, 108, 1, 1.5, 2, 0, 0;
53, 1, 3, 145, 518, 0, 0, 130, 0, 0, 0, 0, 0;
52, 1, 4, 160, 246, 0, 1, 82, 1, 4, 2, 0, 0;
43, 1, 4, 120, 175, 0, 0, 120, 1, 1, 2, 0, 7; // Incluye valor 7 en col 12 y  3
56, 1, 4, 150, 213, 1, 0, 125, 1, 1, 2, 0, 0;

// Muestras con etiqueta 4
54, 0, 3, 130, 294, 0, 1, 100, 1, 0, 2, 0, 0; // Nota: Incluye  4
47, 0, 4, 120, 205, 0, 0, 98, 1, 2, 2, 0, 6; // Nota: Incluye valor 6 en col 12 y  4
52, 1, 4, 140, 266, 0, 0, 134, 1, 2, 2, 0, 0; // Nota: Incluye  4
49, 1, 4, 128, 212, 0, 0, 96, 1, 0, 0, 0, 0; // Nota: Incluye  4
44, 1, 4, 135, 491, 0, 0, 135, 0, 0, 0, 0, 0 // Nota: Incluye  4
]'; // Transpone para que las muestras sean columnas (13x20)

// Etiquetas One-Hotpara las 20 muestras de X_demo.
Y_demo = [
// Etiquetas One-Hot (5 columnas)
1, 0, 0, 0, 0; // Clase 0
1, 0, 0, 0, 0; // Clase 0
1, 0, 0, 0, 0; // Clase 0
1, 0, 0, 0, 0; // Clase 0
1, 0, 0, 0, 0; // Clase 0

0, 1, 0, 0, 0; // Clase 1
0, 1, 0, 0, 0; // Clase 1
0, 1, 0, 0, 0; // Clase 1
0, 1, 0, 0, 0; // Clase 1
0, 1, 0, 0, 0; // Clase 1

0, 0, 1, 0, 0; // Clase 2
0, 0, 1, 0, 0; // Clase 2
0, 0, 1, 0, 0; // Clase 2
0, 0, 1, 0, 0; // Clase 2
0, 0, 1, 0, 0; // Clase 2

0, 0, 0, 1, 0; // Clase 3
0, 0, 0, 1, 0; // Clase 3
0, 0, 0, 1, 0; // Clase 3
0, 0, 0, 1, 0; // Clase 3
0, 0, 0, 1, 0; // Clase 3

0, 0, 0, 0, 1; // Clase 4
0, 0, 0, 0, 1; // Clase 4
0, 0, 0, 0, 1; // Clase 4
0, 0, 0, 0, 1; // Clase 4
0, 0, 0, 0, 1  // Clase 4
]'; // Transpone para que las muestras sean columnas 5 x 20


// Muestra el número de datos de ejemplo usados
n_demo_samples = size(X_demo, 'c');
disp(sprintf('Usando %d muestras de ejemplo', n_demo_samples));

// --- Inicializar la red neuronal ---
// Inicializa pesos y sesgos aleatoriamente. Requiere toolbox ANN.
rand('seed', 44); // Semilla para reproducibilidad
W = ann_FF_init(N); // Esta línea requiere que el toolbox esté cargado

// --- Parámetros de Entrenamiento ---
// Define la tasa de aprendizaje y el número de épocas.
learning_rate = [0.1, 0]; // Tasa de aprendizaje
epochs = 1000;           // Número de épocas para el entrenamiento

// --- Entrenar la red neuronal ---
// Entrena la red usando el algoritmo de retropropagación. Requiere toolbox ANN.
disp(sprintf("Iniciando entrenamiento de demostración con %d épocas...", epochs));
W_trained = ann_FF_Std_online(X_demo, Y_demo, N, W, learning_rate, epochs);
disp("Entrenamiento de demostración completado."); // Solo se mostrará si el entrenamiento termina

// --- Probar con Nuevas Muestras de Ejemplo ---
// Define un pequeño conjunto de datos de prueba hardcodeados (ejemplos no usados en entrenamiento si es posible).
X_test_demo = [
// Muestras de prueba de ejemplo (3 muestras)
42, 1, 2, 120, 198, 0, 0, 155, 0, 0, 0, 0, 0; // Muestra similar a Clase 0
55, 0, 2, 130, 394, 0, 2, 150, 0, 0, 0, 0, 0; // Muestra similar a Clase 0
52, 1, 4, 140, 266, 0, 0, 134, 1, 2, 2, 0, 0; // Muestra similar a Clase 4 (está en set entrenamiento)
47, 1, 3, 140, 193, 0, 0, 145, 1, 1, 2, 0, 0; // Muestra similar a Clase 1
52, 1, 4, 130, 298, 0, 0, 110, 1, 1, 2, 0, 0; // Muestra similar a Clase 3
]'; // Transpone para que las muestras sean columnas (13x5)

// --- Realizar Predicciones ---
// Ejecuta la red entrenada sobre las muestras de prueba. Requiere toolbox ANN.
disp("Realizando predicciones de demostración con ann_FF_run...");
Y_pred_raw_T = ann_FF_run(X_test_demo, N, W_trained);

// --- Interpretar Predicciones y Mostrar Resultados ---
// Interpreta la salida cruda para obtener la clase predicha (0-4).
// Para one-hot, es la neurona con el valor más alto.
predicted_classes_one_based = max(Y_pred_raw_T, 'c'); // Índice (1-based) de la neurona con valor máximo
predicted_classes = predicted_classes_one_based - 1; // Convierte a etiqueta original (0-based)

// Muestra los resultados de las predicciones.
disp("--- Predicciones de Demostración (Datos de Prueba) ---");
disp("Salida Cruda de la Red:");
disp(Y_pred_raw_T);
disp("Clase Predicha (0-4):");
// Muestra las clases predichas como un vector columna.
disp(predicted_classes');
