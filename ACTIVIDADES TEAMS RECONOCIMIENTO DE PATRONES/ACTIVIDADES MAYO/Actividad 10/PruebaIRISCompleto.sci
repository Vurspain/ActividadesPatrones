//Alumno Carlos Alejandro Acosta Mendez
//red neuronal de 4 entradas 10 neuronas ocultas y 3 salidas
//para predecir las clases de la flor IRIS 
//Se divide el dataset en entrenamiento y prueba.

// --- Configuración de la Red ---
n_entradas = 4;    // Número de entradas 
n_ocultas = 10;    // Neuronas en la capa oculta
n_salidas = 3;     // Neuronas en la capa de salida (corresponde a 3 clases 0, 1, 2)

// --- Funciones de Activación ---
// Función sigmoide
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
endfunction

// Derivada de la sigmoide
function y = sigmoid_derivada(x)
    y = sigmoid(x) .* (1 - sigmoid(x));
endfunction

// --- Datos IRIS completos (con etiquetas 0, 1, 2) ---
// Las 4 primeras columnas son features, la 5ta es la etiqueta (0, 1, 2)
IRIS_DATA=[
    [5.1,3.5,1.4,0.2,0],[4.9,3,1.4,0.2,0],[4.7,3.2,1.3,0.2,0],
    [4.6,3.1,1.5,0.2,0],[5,3.6,1.4,0.2,0],[5.4,3.9,1.7,0.4,0],
    [4.6,3.4,1.4,0.3,0],[5,3.4,1.5,0.2,0],[4.4,2.9,1.4,0.2,0],
    [4.9,3.1,1.5,0.1,0],[5.4,3.7,1.5,0.2,0],[4.8,3.4,1.6,0.2,0],
    [4.8,3,1.4,0.1,0],[4.3,3,1.1,0.1,0],[5.8,4,1.2,0.2,0],
    [5.7,4.4,1.5,0.4,0],[5.4,3.9,1.3,0.4,0],[5.1,3.5,1.4,0.3,0],
    [5.7,3.8,1.7,0.3,0],[5.1,3.8,1.5,0.3,0],[5.4,3.4,1.7,0.2,0],
    [5.1,3.7,1.5,0.4,0],[4.6,3.6,1,0.2,0],[5.1,3.3,1.7,0.5,0],
    [4.8,3.4,1.9,0.2,0],[5,3,1.6,0.2,0],[5,3.4,1.6,0.4,0],
    [5.2,3.5,1.5,0.2,0],[5.2,3.4,1.4,0.2,0],[4.7,3.2,1.6,0.2,0],
    [4.8,3.1,1.6,0.2,0],[5.4,3.4,1.5,0.4,0],[5.2,4.1,1.5,0.1,0],
    [5.5,4.2,1.4,0.2,0],[4.9,3.1,1.5,0.1,0],[5,3.2,1.2,0.2,0],
    [5.5,3.5,1.3,0.2,0],[4.9,3.1,1.5,0.1,0],[4.4,3,1.3,0.2,0],
    [5.1,3.4,1.5,0.2,0],[5,3.5,1.3,0.3,0],[4.5,2.3,1.3,0.3,0],
    [4.4,3.2,1.3,0.2,0],[5,3.5,1.6,0.6,0],[5.1,3.8,1.9,0.4,0],
    [4.8,3,1.4,0.3,0],[5.1,3.8,1.6,0.2,0],[4.6,3.2,1.4,0.2,0],
    [5.3,3.7,1.5,0.2,0],[5,3.3,1.4,0.2,0],[7,3.2,4.7,1.4,1],
    [6.4,3.2,4.5,1.5,1],[6.9,3.1,4.9,1.5,1],[5.5,2.3,4,1.3,1],
    [6.5,2.8,4.6,1.5,1],[5.7,2.8,4.5,1.3,1],[6.3,3.3,4.7,1.6,1],
    [4.9,2.4,3.3,1,1],[6.6,2.9,4.6,1.3,1],[5.2,2.7,3.9,1.4,1],
    [5,2,3.5,1,1],[5.9,3,4.2,1.5,1],[6,2.2,4,1,1],
    [6.1,2.9,4.7,1.4,1],[5.6,2.9,3.6,1.3,1],[6.7,3.1,4.4,1.4,1],
    [5.6,3,4.5,1.5,1],[5.8,2.7,4.1,1,1],[6.2,2.2,4.5,1.5,1],
    [5.6,2.5,3.9,1.1,1],[5.9,3.2,4.8,1.8,1],[6.1,2.8,4,1.3,1],
    [6.3,2.5,4.9,1.5,1],[6.1,2.8,4.7,1.2,1],[6.4,2.9,4.3,1.3,1],
    [6.6,3,4.4,1.4,1],[6.8,2.8,4.8,1.4,1],[6.7,3,5,1.7,1],
    [6,2.9,4.5,1.5,1],[5.7,2.6,3.5,1,1],[5.5,2.4,3.8,1.1,1],
    [5.5,2.4,3.7,1,1],[5.8,2.7,3.9,1.2,1],[6,2.7,5.1,1.6,1],
    [5.4,3,4.5,1.5,1],[6,3.4,4.5,1.6,1],[6.7,3.1,4.7,1.5,1],
    [6.3,2.3,4.4,1.3,1],[5.6,3,4.1,1.3,1],[5.5,2.5,4,1.3,1],
    [5.5,2.6,4.4,1.2,1],[6.1,3,4.6,1.4,1],[5.8,2.6,4,1.2,1],
    [5,2.3,3.3,1,1],[5.6,2.7,4.2,1.3,1],[5.7,3,4.2,1.2,1],
    [5.7,2.9,4.2,1.3,1],[6.2,2.9,4.3,1.3,1],[5.1,2.5,3,1.1,1],
    [5.7,2.8,4.1,1.3,1],[6.3,3.3,6,2.5,2],[5.8,2.7,5.1,1.9,2],
    [7.1,3,5.9,2.1,2],[6.3,2.9,5.6,1.8,2],[6.5,3,5.8,2.2,2],
    [7.6,3,6.6,2.1,2],[4.9,2.5,4.5,1.7,2],[7.3,2.9,6.3,1.8,2],
    [6.7,2.5,5.8,1.8,2],[7.2,3.6,6.1,2.5,2],[6.5,3.2,5.1,2,2],
    [6.4,2.7,5.3,1.9,2],[6.8,3,5.5,2.1,2],[5.7,2.5,5,2,2],
    [5.8,2.8,5.1,2.4,2],[6.4,3.2,5.3,2.3,2],[6.5,3,5.5,1.8,2],
    [7.7,3.8,6.7,2.2,2],[7.7,2.6,6.9,2.3,2],[6,2.2,5,1.5,2],
    [6.9,3.2,5.7,2.3,2],[5.6,2.8,4.9,2,2],[7.7,2.8,6.7,2,2],
    [6.3,2.7,4.9,1.8,2],[6.7,3.3,5.7,2.1,2],[7.2,3.2,6,1.8,2],
    [6.2,2.8,4.8,1.8,2],[6.1,3,4.9,1.8,2],[6.4,2.8,5.6,2.1,2],
    [7.2,3,5.8,1.6,2],[7.4,2.8,6.1,1.9,2],[7.9,3.8,6.4,2,2],
    [6.4,2.8,5.6,2.2,2],[6.3,2.8,5.1,1.5,2],[6.1,2.6,5.6,1.4,2],
    [7.7,3,6.1,2.3,2],[6.3,3.4,5.6,2.4,2],[6.4,3.1,5.5,1.8,2],
    [6,3,4.8,1.8,2],[6.9,3.1,5.4,2.1,2],[6.7,3.1,5.6,2.4,2],
    [6.9,3.1,5.1,2.3,2],[5.8,2.7,5.1,1.9,2],[6.8,3.2,5.9,2.3,2],
    [6.7,3.3,5.7,2.5,2],[6.7,3,5.2,2.3,2],[6.3,2.5,5,1.9,2],
    [6.5,3,5.2,2,2],[6.2,3.4,5.4,2.3,2],[5.9,3,5.1,1.8,2]
];

// --- Separar características y etiquetas ---
X_all = IRIS_DATA(:, 1:n_entradas); // Las 4 columnas de características
Y_all_numeric = IRIS_DATA(:, n_entradas + 1); // La columna de etiquetas numéricas (0, 1, 2)

n_total_samples = size(X_all, 'r'); // Número total de observaciones 
disp(sprintf('Dataset IRIS cargado con %d observaciones.', n_total_samples));

// --- Convertir etiquetas numéricas (0, 1, 2) a One-Hot Encoding ---
// 0 -> [1,0,0], 1 -> [0,1,0], 2 -> [0,0,1]
Y_all_onehot = zeros(n_total_samples, n_salidas);
for i = 1:n_total_samples
    class_index = Y_all_numeric(i); // Las etiquetas ya son 0, 1, 2
    if class_index >= 0 && class_index < n_salidas then
         Y_all_onehot(i, class_index + 1) = 1; // Usamos class_index + 1 para el índice en Scilab (1-based)
    else
         error(sprintf('Etiqueta de clase inesperada en la fila %d: %f', i, Y_all_numeric(i)));
    end
end

// --- Dividir los datos en conjuntos de entrenamiento y prueba ---
split_ratio = 0.8; // 80% para entrenamiento, 20% para prueba
n_train_samples = floor(split_ratio * n_total_samples);
n_test_samples = n_total_samples - n_train_samples;

// Generar índices aleatorios para la división
rand('seed', 42); // Usar la misma semilla para la división si quieres reproducibilidad

// gsort con números aleatorios para obtener una permutación ---
// gsort retorna [sorted_array, permutation_indices]
// Genera n_total_samples números aleatorios, los ordena GLOBALMENTE ('g')
// y retorna los índices originales de esos números ordenados (la permutación).
[sorted_rand_values_dummy, rand_indices] = gsort(rand(1, n_total_samples), 'g');


// Obtener índices para entrenamiento y prueba
train_indices = rand_indices(1:n_train_samples);
test_indices = rand_indices(n_train_samples+1:$);

// Crear conjuntos de entrenamiento
X_train = X_all(train_indices, :);
Y_train = Y_all_onehot(train_indices, :);
Y_train_numeric = Y_all_numeric(train_indices, :); // Guardar también las etiquetas numéricas si es necesario

// Crear conjuntos de prueba
X_test = X_all(test_indices, :);
Y_test = Y_all_onehot(test_indices, :);
Y_test_numeric = Y_all_numeric(test_indices, :); // Guardar también las etiquetas numéricas para evaluación

disp(sprintf('Datos divididos: %d para entrenamiento, %d para prueba.', n_train_samples, n_test_samples));


// --- Inicializar pesos y sesgos aleatoriamente ---
// Los pesos y sesgos se inicializan antes del entrenamiento
rand('seed', 43); // Usar una semilla diferente para la inicialización si quieres
W1 = rand(n_entradas, n_ocultas) * 0.01; // Pesos capa oculta (inicialización pequeña)
b1 = rand(1, n_ocultas) * 0.01;          // Sesgos capa oculta (inicialización pequeña)
W2 = rand(n_ocultas, n_salidas) * 0.01;  // Pesos capa salida (inicialización pequeña)
b2 = rand(1, n_salidas) * 0.01;          // Sesgos capa salida (inicialización pequeña)

// --- Entrenamiento de la Red ---
disp("Iniciando entrenamiento...");

// Hiperparámetros
tasa_aprendizaje = 0.1;
max_iter = 5000; // Aumentar iteraciones para entrenar con más datos

// Entrenamiento
for iter = 1:max_iter
    // Propagación hacia adelante (usando datos de entrenamiento)
    // Expandir sesgos para sumarlos a cada muestra en el lote
    b1_expanded = repmat(b1, n_train_samples, 1);
    Z1 = X_train * W1 + b1_expanded;
    A1 = sigmoid(Z1); // Salidas de la capa oculta

    b2_expanded = repmat(b2, n_train_samples, 1);
    Z2 = A1 * W2 + b2_expanded;
    A2 = sigmoid(Z2); // Salidas de la capa de salida (predicciones)

    // Cálculo del error (Error Cuadrático Medio implícito en la retropropagación)
    error = Y_train - A2; // Diferencia entre la etiqueta real y la predicha

    // Retropropagación
    // Gradiente en la capa de salida
    dZ2 = error .* sigmoid_derivada(Z2); // Error multiplicado por la derivada de la activación
    dW2 = A1' * dZ2;                     // Gradiente de los pesos de salida
    db2 = sum(dZ2, 1);                   // Gradiente de los sesgos de salida (sumar a través de las muestras)

    // Gradiente en la capa oculta
    dZ1 = (dZ2 * W2') .* sigmoid_derivada(Z1); // Error propagado a la capa oculta * derivada de la activación oculta
    dW1 = X_train' * dZ1;                     // Gradiente de los pesos ocultos
    db1 = sum(dZ1, 1);                        // Gradiente de los sesgos ocultos

    // Actualizar pesos y sesgos (Descenso de Gradiente)
    W2 = W2 + tasa_aprendizaje * dW2;
    b2 = b2 + tasa_aprendizaje * db2;
    W1 = W1 + tasa_aprendizaje * dW1;
    b1 = b1 + tasa_aprendizaje * db1;

    // Mostrar el error cada cierto número de iteraciones (opcional)
    if modulo(iter, 500) == 0 then
        // Calcular el error cuadrático medio total
        mse = mean(mean((Y_train - A2).^2));
        disp(sprintf('Iteración %d, MSE de Entrenamiento: %f', iter, mse));
    end
end

disp("Entrenamiento completado.");

// --- Evaluación en el Conjunto de Prueba ---
disp("Evaluando en el conjunto de prueba...");

// Propagación hacia adelante con datos de prueba
b1_test_expanded = repmat(b1, n_test_samples, 1);
Z1_test = X_test * W1 + b1_test_expanded;
A1_test = sigmoid(Z1_test);

b2_test_expanded = repmat(b2, n_test_samples, 1);
Z2_test = A1_test * W2 + b2_test_expanded;
Y_pred_test = sigmoid(Z2_test); // Salidas de la red para los datos de prueba (probabilidades para clases 0, 1, 2)

// Convertir las salidas (probabilidades) a clases predichas (0, 1, o 2)
predicted_classes = zeros(n_test_samples, 1);
for i = 1:n_test_samples
    // Encontrar el índice (1-based) de la probabilidad máxima en la fila i
    [max_prob, class_idx_one_based] = max(Y_pred_test(i, :));
    // Convertir el índice 1-based a la etiqueta numérica 0, 1, o 2
    predicted_classes(i) = class_idx_one_based - 1;
end

// Comparar predicciones con las etiquetas reales del conjunto de prueba (que ya son 0, 1, 2)
actual_classes = Y_test_numeric;

// Calcular precisión (accuracy)
correct_predictions = sum(predicted_classes == actual_classes);
accuracy = correct_predictions / n_test_samples * 100;

disp("Resultados en el conjunto de prueba:");
disp(sprintf('Número de predicciones correctas: %d', correct_predictions));
disp(sprintf('Número total de muestras de prueba: %d', n_test_samples));
disp(sprintf('Precisión (Accuracy): %.2f%%', accuracy));

// Mostrar algunas predicciones y valores reales
disp("Ejemplos de predicciones vs Reales (del conjunto de prueba):");
disp("Predicho | Real");
disp("-----------------");
for i = 1:min(10, n_test_samples) // Mostrar los primeros 10 o menos si hay menos muestras
    disp(sprintf('  %d      |  %d', predicted_classes(i), actual_classes(i)));
end
