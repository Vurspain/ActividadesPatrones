// Alumno Carlos Alejandro Acosta Mendez
// Red neuronal para clasificación de la flor IRIS usando el toolbox ANN
// Estructura: 4 entradas, 10 neuronas ocultas, 3 salidas (para 3 clases)
// Se añade escalado de datos para mejorar el entrenamiento.

// --- Cargar el ANN Toolbox ---
atomsLoad("ANN_Toolbox");

// --- Configuración de la Red ---
n_entradas = 4;    // Número de entradas (características de IRIS)
n_ocultas = 10;    // Neuronas en la capa oculta
n_salidas = 3;     // Neuronas en la capa de salida (corresponde a 3 clases: Setosa, Versicolor, Virginica)

N = [n_entradas, n_ocultas, n_salidas];

// --- Datos IRIS completos (con etiquetas 0, 1, 2) ---
// Las 4 primeras columnas son features, la 5ta es la etiqueta (0, 1, 2)
// Este bloque debe contener las 150 filas del dataset
// --- Datos IRIS completos (con etiquetas 0, 1, 2) ---
// Las 4 primeras columnas son features, la 5ta es la etiqueta (0, 1, 2)
// Este bloque debe contener las 150 filas del dataset
// --- Datos IRIS completos (150 muestras, 5 columnas: 4 features + 1 etiqueta) ---
// Cada fila '[...]' es una muestra. Las filas están separadas por ';', excepto la última.
// Los valores dentro de '[...]' están separados por ','.
IRIS_DATA=[
    [5.1,3.5,1.4,0.2,0]; // Muestra 1
    [4.9,3,1.4,0.2,0]; // Muestra 2
    [4.7,3.2,1.3,0.2,0]; // Muestra 3
    [4.6,3.1,1.5,0.2,0]; // Muestra 4
    [5,3.6,1.4,0.2,0]; // Muestra 5
    [5.4,3.9,1.7,0.4,0]; // Muestra 6
    [4.6,3.4,1.4,0.3,0]; // Muestra 7
    [5,3.4,1.5,0.2,0]; // Muestra 8
    [4.4,2.9,1.4,0.2,0]; // Muestra 9
    [4.9,3.1,1.5,0.1,0]; // Muestra 10
    [5.4,3.7,1.5,0.2,0]; // Muestra 11
    [4.8,3.4,1.6,0.2,0]; // Muestra 12
    [4.8,3,1.4,0.1,0]; // Muestra 13
    [4.3,3,1.1,0.1,0]; // Muestra 14
    [5.8,4,1.2,0.2,0]; // Muestra 15
    [5.7,4.4,1.5,0.4,0]; // Muestra 16
    [5.4,3.9,1.3,0.4,0]; // Muestra 17
    [5.1,3.5,1.4,0.3,0]; // Muestra 18
    [5.7,3.8,1.7,0.3,0]; // Muestra 19
    [5.1,3.8,1.5,0.3,0]; // Muestra 20
    [5.4,3.4,1.7,0.2,0]; // Muestra 21
    [5.1,3.7,1.5,0.4,0]; // Muestra 22
    [4.6,3.6,1,0.2,0]; // Muestra 23
    [5.1,3.3,1.7,0.5,0]; // Muestra 24
    [4.8,3.4,1.9,0.2,0]; // Muestra 25
    [5,3,1.6,0.2,0]; // Muestra 26
    [5,3.4,1.6,0.4,0]; // Muestra 27
    [5.2,3.5,1.5,0.2,0]; // Muestra 28
    [5.2,3.4,1.4,0.2,0]; // Muestra 29
    [4.7,3.2,1.6,0.2,0]; // Muestra 30
    [4.8,3.1,1.6,0.2,0]; // Muestra 31
    [5.4,3.4,1.5,0.4,0]; // Muestra 32
    [5.2,4.1,1.5,0.1,0]; // Muestra 33
    [5.5,4.2,1.4,0.2,0]; // Muestra 34
    [4.9,3.1,1.5,0.1,0]; // Muestra 35
    [5,3.2,1.2,0.2,0]; // Muestra 36
    [5.5,3.5,1.3,0.2,0]; // Muestra 37
    [4.9,3.1,1.5,0.1,0]; // Muestra 38
    [4.4,3,1.3,0.2,0]; // Muestra 39
    [5.1,3.4,1.5,0.2,0]; // Muestra 40
    [5,3.5,1.3,0.3,0]; // Muestra 41
    [4.5,2.3,1.3,0.3,0]; // Muestra 42
    [4.4,3.2,1.3,0.2,0]; // Muestra 43
    [5,3.5,1.6,0.6,0]; // Muestra 44
    [5.1,3.8,1.9,0.4,0]; // Muestra 45
    [4.8,3,1.4,0.3,0]; // Muestra 46
    [5.1,3.8,1.6,0.2,0]; // Muestra 47
    [4.6,3.2,1.4,0.2,0]; // Muestra 48
    [5.3,3.7,1.5,0.2,0]; // Muestra 49
    [5,3.3,1.4,0.2,0]; // Muestra 50
    [7,3.2,4.7,1.4,1]; // Muestra 51
    [6.4,3.2,4.5,1.5,1]; // Muestra 52
    [6.9,3.1,4.9,1.5,1]; // Muestra 53
    [5.5,2.3,4,1.3,1]; // Muestra 54
    [6.5,2.8,4.6,1.5,1]; // Muestra 55
    [5.7,2.8,4.5,1.3,1]; // Muestra 56
    [6.3,3.3,4.7,1.6,1]; // Muestra 57
    [4.9,2.4,3.3,1,1]; // Muestra 58
    [6.6,2.9,4.6,1.3,1]; // Muestra 59
    [5.2,2.7,3.9,1.4,1]; // Muestra 60
    [5,2,3.5,1,1]; // Muestra 61
    [5.9,3,4.2,1.5,1]; // Muestra 62
    [6,2.2,4,1,1]; // Muestra 63
    [6.1,2.9,4.7,1.4,1]; // Muestra 64
    [5.6,2.9,3.6,1.3,1]; // Muestra 65
    [6.7,3.1,4.4,1.4,1]; // Muestra 66
    [5.6,3,4.5,1.5,1]; // Muestra 67
    [5.8,2.7,4.1,1,1]; // Muestra 68
    [6.2,2.2,4.5,1.5,1]; // Muestra 69
    [5.6,2.5,3.9,1.1,1]; // Muestra 70
    [5.9,3.2,4.8,1.8,1]; // Muestra 71
    [6.1,2.8,4,1.3,1]; // Muestra 72
    [6.3,2.5,4.9,1.5,1]; // Muestra 73
    [6.1,2.8,4.7,1.2,1]; // Muestra 74
    [6.4,2.9,4.3,1.3,1]; // Muestra 75
    [6.6,3,4.4,1.4,1]; // Muestra 76
    [6.8,2.8,4.8,1.4,1]; // Muestra 77
    [6.7,3,5,1.7,1]; // Muestra 78
    [6,2.9,4.5,1.5,1]; // Muestra 79
    [5.7,2.6,3.5,1,1]; // Muestra 80
    [5.5,2.4,3.8,1.1,1]; // Muestra 81
    [5.5,2.4,3.7,1,1]; // Muestra 82
    [5.8,2.7,3.9,1.2,1]; // Muestra 83
    [6,2.7,5.1,1.6,1]; // Muestra 84
    [5.4,3,4.5,1.5,1]; // Muestra 85
    [6,3.4,4.5,1.6,1]; // Muestra 86
    [6.7,3.1,4.7,1.5,1]; // Muestra 87
    [6.3,2.3,4.4,1.3,1]; // Muestra 88
    [5.6,3,4.1,1.3,1]; // Muestra 89
    [5.5,2.5,4,1.3,1]; // Muestra 90
    [5.5,2.6,4.4,1.2,1]; // Muestra 91
    [6.1,3,4.6,1.4,1]; // Muestra 92
    [5.8,2.6,4,1.2,1]; // Muestra 93
    [5,2.3,3.3,1,1]; // Muestra 94
    [5.6,2.7,4.2,1.3,1]; // Muestra 95
    [5.7,3,4.2,1.2,1]; // Muestra 96
    [5.7,2.9,4.2,1.3,1]; // Muestra 97
    [6.2,2.9,4.3,1.3,1]; // Muestra 98
    [5.1,2.5,3,1.1,1]; // Muestra 99
    [5.7,2.8,4.1,1.3,1]; // Muestra 100
    [6.3,3.3,6,2.5,2]; // Muestra 101
    [5.8,2.7,5.1,1.9,2]; // Muestra 102
    [7.1,3,5.9,2.1,2]; // Muestra 103
    [6.3,2.9,5.6,1.8,2]; // Muestra 104
    [6.5,3,5.8,2.2,2]; // Muestra 105
    [7.6,3,6.6,2.1,2]; // Muestra 106
    [4.9,2.5,4.5,1.7,2]; // Muestra 107
    [7.3,2.9,6.3,1.8,2]; // Muestra 108
    [6.7,2.5,5.8,1.8,2]; // Muestra 109
    [7.2,3.6,6.1,2.5,2]; // Muestra 110
    [6.5,3.2,5.1,2,2]; // Muestra 111
    [6.4,2.7,5.3,1.9,2]; // Muestra 112
    [6.8,3,5.5,2.1,2]; // Muestra 113
    [5.7,2.5,5,2,2]; // Muestra 114
    [5.8,2.8,5.1,2.4,2]; // Muestra 115
    [6.4,3.2,5.3,2.3,2]; // Muestra 116
    [6.5,3,5.5,1.8,2]; // Muestra 117
    [7.7,3.8,6.7,2.2,2]; // Muestra 118
    [7.7,2.6,6.9,2.3,2]; // Muestra 119
    [6,2.2,5,1.5,2]; // Muestra 120
    [6.9,3.2,5.7,2.3,2]; // Muestra 121
    [5.6,2.8,4.9,2,2]; // Muestra 122
    [7.7,2.8,6.7,2,2]; // Muestra 123
    [6.3,2.7,4.9,1.8,2]; // Muestra 124
    [6.7,3.3,5.7,2.1,2]; // Muestra 125
    [7.2,3.2,6,1.8,2]; // Muestra 126
    [6.2,2.8,4.8,1.8,2]; // Muestra 127
    [6.1,3,4.9,1.8,2]; // Muestra 128
    [6.4,2.8,5.6,2.1,2]; // Muestra 129
    [7.2,3,5.8,1.6,2]; // Muestra 130
    [7.4,2.8,6.1,1.9,2]; // Muestra 131
    [7.9,3.8,6.4,2,2]; // Muestra 132
    [6.4,2.8,5.6,2.2,2]; // Muestra 133
    [6.3,2.8,5.1,1.5,2]; // Muestra 134
    [6.1,2.6,5.6,1.4,2]; // Muestra 135
    [7.7,3,6.1,2.3,2]; // Muestra 136
    [6.3,3.4,5.6,2.4,2]; // Muestra 137
    [6.4,3.1,5.5,1.8,2]; // Muestra 138
    [6,3,4.8,1.8,2]; // Muestra 139
    [6.9,3.1,5.4,2.1,2]; // Muestra 140
    [6.7,3.1,5.6,2.4,2]; // Muestra 141
    [6.9,3.1,5.1,2.3,2]; // Muestra 142
    [5.8,2.7,5.1,1.9,2]; // Muestra 143
    [6.8,3.2,5.9,2.3,2]; // Muestra 144
    [6.7,3.3,5.7,2.5,2]; // Muestra 145
    [6.7,3,5.2,2.3,2]; // Muestra 146
    [6.3,2.5,5,1.9,2]; // Muestra 147
    [6.5,3,5.2,2,2]; // Muestra 148
    [6.2,3.4,5.4,2.3,2]; // Muestra 149
    [5.9,3,5.1,1.8,2]  // Muestra 150 
];


// --- Separar características y etiquetas ---
X_all = IRIS_DATA(:, 1:n_entradas); // Las 4 columnas de características (150x4)
Y_all_numeric = IRIS_DATA(:, n_entradas + 1); // La columna de etiquetas numéricas (150x1)
n_total_samples = size(X_all, 'r'); // Número total de observaciones (150)
disp(sprintf('Dataset IRIS cargado con %d observaciones.', n_total_samples));

// --- Escalar las características de entrada (Min-Max Scaling a [0, 1]) ---
// Esto ayuda al entrenamiento de la red neuronal
X_all_scaled = zeros(size(X_all));
for j = 1:n_entradas
    min_val = min(X_all(:, j));
    max_val = max(X_all(:, j));
    // Evitar división por cero si la característica es constante
    if max_val - min_val > 0 then
        X_all_scaled(:, j) = (X_all(:, j) - min_val) / (max_val - min_val);
    else
        // Si la característica es constante, simplemente mantenerla (o ponerla a 0.5 si se quiere)
        X_all_scaled(:, j) = X_all(:, j); // O podrías poner zeros(size(X_all(:, j))) + 0.5;
    end
end
// Usaremos X_all_scaled en lugar de X_all en adelante
disp("Características de entrada escaladas (Min-Max).");


// --- Convertir etiquetas numéricas (0, 1, 2) a One-Hot Encoding ---
Y_all_onehot = zeros(n_total_samples, n_salidas); // Matriz de 150x3
for i = 1:n_total_samples
    class_index = Y_all_numeric(i); // Las etiquetas ya son 0, 1, 2
    if class_index >= 0 && class_index < n_salidas then
         Y_all_onehot(i, class_index + 1) = 1; // Usamos class_index + 1 para el índice en Scilab (1-based)
    else
         // Esto no debería ocurrir con los datos IRIS estándar
         error(sprintf('Etiqueta de clase inesperada en la fila %d: %f', i, Y_all_numeric(i)));
    end
end

// --- Transponer los datos para que las muestras sean columnas (formato del toolbox ANN) ---
// Usamos los datos escalados
X_all_T = X_all_scaled'; // Matriz 4x150
Y_all_onehot_T = Y_all_onehot'; // Matriz 3x150

// --- Dividir los datos en conjuntos de entrenamiento y prueba ---
split_ratio = 0.8; // 80% para entrenamiento, 20% para prueba
n_total_samples_T = size(X_all_T, 'c'); // Número total de muestras (150)
n_train_samples_T = floor(split_ratio * n_total_samples_T);
n_test_samples_T = n_total_samples_T - n_train_samples_T;

rand('seed', 42); // Usar la misma semilla para la división si quieres reproducibilidad
[sorted_rand_values_dummy, rand_indices] = gsort(rand(1, n_total_samples_T), 'g');

train_indices = rand_indices(1:n_train_samples_T);
test_indices = rand_indices(n_train_samples_T+1:$);

// Crear conjuntos de entrenamiento y prueba (usando datos escalados y transpuestos)
X_train_T = X_all_T(:, train_indices); // Matriz 4x120 (aprox)
Y_train_T = Y_all_onehot_T(:, train_indices); // Matriz 3x120 (aprox)

X_test_T = X_all_T(:, test_indices); // Matriz 4x30 (aprox)
Y_test_T = Y_all_onehot_T(:, test_indices); // Matriz 3x30 (aprox)
Y_test_numeric = Y_all_numeric(test_indices); // Vector 30x1 (aprox)

disp(sprintf('Datos divididos: %d para entrenamiento, %d para prueba.', n_train_samples_T, n_test_samples_T));

// --- Inicializar la red neuronal ---
rand('seed', 43);
W = ann_FF_init(N); // Inicializa la estructura de pesos y sesgos W

// --- Parámetros de Entrenamiento ---
learning_rate = [0.1, 0]; // Tasa de aprendizaje
epochs = 5000;           // Vuelve a un número de épocas adecuado

// --- Entrenar la red ---
disp(sprintf("Iniciando entrenamiento con ann_FF_Std_online para %d épocas...", epochs));
W_trained = ann_FF_Std_online(X_train_T, Y_train_T, N, W, learning_rate, epochs);
disp("Entrenamiento completado.");

// --- Evaluación en el Conjunto de Prueba ---
disp("Evaluando en el conjunto de prueba con ann_FF_run...");
Y_pred_raw_T = ann_FF_run(X_test_T, N, W_trained); // Matriz (outputs x test_samples)

// --- Convertir las salidas de la red a clases predichas (0, 1, o 2) ---
predicted_classes_one_based = zeros(1, n_test_samples_T);
[max_probs, predicted_classes_one_based] = max(Y_pred_raw_T, 'c');

// Convertir índices 1-based a etiquetas 0-based
predicted_classes = predicted_classes_one_based - 1; // Vector fila 1x(num_test_samples)

// Comparar predicciones con etiquetas reales
// Y_test_numeric es vector columna (num_test_samples)x1
// predicted_classes es vector fila 1x(num_test_samples)
correct_predictions = sum(predicted_classes' == Y_test_numeric);
accuracy = correct_predictions / n_test_samples_T * 100;

disp("--- Resultados en el conjunto de prueba ---");
disp(sprintf('Número de predicciones correctas: %d', correct_predictions));
disp(sprintf('Número total de muestras de prueba: %d', n_test_samples_T));
disp(sprintf('Precisión (Accuracy): %.2f%%', accuracy));

// --- Mostrar algunas predicciones y valores reales ---
disp("--- Ejemplos de predicciones vs Reales (del conjunto de prueba) ---");
disp("Predicho | Real");
disp("-----------------");
predicted_classes_col = predicted_classes'; // Transponer para mostrar como columna
for i = 1:min(10, n_test_samples_T) // Mostrar los primeros 10 o menos
    // Usamos int() para asegurar que los valores pasados a %d sean enteros y evitar el error
    disp(sprintf('  %d      |  %d', int(predicted_classes_col(i)), int(Y_test_numeric(i)))); // <-- CORRECCIÓN
end
