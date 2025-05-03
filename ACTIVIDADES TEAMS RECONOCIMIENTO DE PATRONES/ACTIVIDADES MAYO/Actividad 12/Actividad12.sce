// =============================================================================
// PARÁMETROS DE LA RED NEURONAL
// =============================================================================
n_entradas = 10;  // Número de entradas reducido a 10
n_ocultas = 8;    // Neuronas en la capa oculta (ajustado)
n_salidas = 3;    // Neuronas en la capa de salida (3 clases)
n_num_dat_ent = 30; // Número de datos de entrenamiento

// =============================================================================
// FUNCIONES DE ACTIVACIÓN Y DERIVADAS
// =============================================================================
// Función de activación sigmoide
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
endfunction

// Derivada de la función sigmoide
function y = sigmoid_derivada(x)
    y = sigmoid(x) .* (1 - sigmoid(x));
endfunction

// =============================================================================
// INICIALIZACIÓN DE PESOS Y SESGOS
// =============================================================================
// Pesos y sesgos de la capa oculta
W1 = rand(n_entradas, n_ocultas);  // Pesos capa oculta
b1 = rand(1, n_ocultas);           // Sesgos capa oculta

// Pesos y sesgos de la capa de salida
W2 = rand(n_ocultas, n_salidas);   // Pesos capa salida
b2 = rand(1, n_salidas);           // Sesgos capa salida

// =============================================================================
// CARGA Y PREPROCESAMIENTO DE DATOS
// =============================================================================
// Ruta del archivo de datos 
file_path = "C:\Users\caam0\OneDrive\Escritorio\SEMESTRE 6\RECONOCIMIENTO DE PATRONES\Actividad 12\reprocessed.hungarian.data";

// Abrir el archivo de datos
data = mopen(file_path, 'r');

// Verificar si el archivo se abrió correctamente
if data == -1 then
    disp("Error: No se puede abrir el archivo.");
    quit;
else
    // Leer todas las líneas del archivo
    raw_data = mgetl(data);
    mclose(data);  // Cerrar el archivo después de leer
end

// Procesar los datos
n_rows = size(raw_data, 'r');  // Número de filas en los datos
parsed_data = [];              // Matriz para almacenar los datos procesados

for i = 1:n_rows
    // Reemplazar comas por espacios y convertir a números
    line = strsubst(raw_data(i), ",", " ");
    parsed_data = [parsed_data; evstr(line)];
end

// Manejar valores faltantes (reemplazar -9 con 0)
parsed_data(parsed_data == -9) = 0;

// Separar características (X) y etiquetas (Y_raw) - ahora solo 10 características
X = parsed_data(:, 1:10);  // Solo las primeras 10 características
Y_raw = parsed_data(:, 14); // Etiquetas (columna 14)

// Codificar las etiquetas en formato one-hot para 3 clases
Y = zeros(size(Y_raw, 1), n_salidas);  // Matriz de etiquetas one-hot

// Asegurar que las etiquetas estén en el rango 0-2
Y_raw(Y_raw > 2) = 2; // Si había valores mayores a 2, los limitamos a 2

for i = 1:size(Y_raw, 1)
    if Y_raw(i) >= 0 & Y_raw(i) <= 2 then
        Y(i, Y_raw(i) + 1) = 1;  // Asignar 1 en la posición correspondiente
    else
        // Si no está en el rango, asignar a una clase por defecto (ej. clase 0)
        Y(i, 1) = 1;
    end
end

// Mostrar datos de entrada y etiquetas
disp("Datos de entrada y etiquetas (one-hot):");
disp(cat(2, X, Y));

// =============================================================================
// ENTRENAMIENTO DE LA RED NEURONAL
// =============================================================================
disp("Entrenamiento de la red neuronal:");

// Hiperparámetros
tasa_aprendizaje = 0.1;  // Tasa de aprendizaje
max_iter = 1000;         // Número máximo de iteraciones

for iter = 1:max_iter
    // =========================================
    // PROPAGACIÓN HACIA ADELANTE (FORWARD PASS)
    // =========================================
    // Capa oculta
    b1_expanded = repmat(b1, size(X, 1), 1);  // Expandir sesgos
    Z1 = X * W1 + b1_expanded;               // Entrada ponderada + sesgo
    A1 = sigmoid(Z1);                        // Salida de la capa oculta

    // Capa de salida
    b2_expanded = repmat(b2, size(X, 1), 1);  // Expandir sesgos
    Z2 = A1 * W2 + b2_expanded;               // Entrada ponderada + sesgo
    A2 = sigmoid(Z2);                         // Salida de la capa de salida

    // =========================================
    // CÁLCULO DEL ERROR
    // =========================================
    error = Y - A2;  // Error entre las salidas predichas y las reales

    // =========================================
    // RETROPROPAGACIÓN (BACKPROPAGATION)
    // =========================================
    // Gradientes de la capa de salida
    dZ2 = error .* sigmoid_derivada(Z2);  // Derivada del error respecto a Z2
    dW2 = A1' * dZ2;                     // Gradiente de los pesos de salida
    db2 = sum(dZ2, 1);                   // Gradiente de los sesgos de salida

    // Gradientes de la capa oculta
    dZ1 = (dZ2 * W2') .* sigmoid_derivada(Z1);  // Derivada del error respecto a Z1
    dW1 = X' * dZ1;                             // Gradiente de los pesos ocultos
    db1 = sum(dZ1, 1);                          // Gradiente de los sesgos ocultos

    // =========================================
    // ACTUALIZACIÓN DE PESOS Y SESGOS
    // =========================================
    W2 = W2 + tasa_aprendizaje * dW2;  // Actualizar pesos de salida
    b2 = b2 + tasa_aprendizaje * db2;  // Actualizar sesgos de salida
    W1 = W1 + tasa_aprendizaje * dW1;  // Actualizar pesos ocultos
    b1 = b1 + tasa_aprendizaje * db1;  // Actualizar sesgos ocultos
end

// =============================================================================
// PRUEBA DE LA RED NEURONAL
// =============================================================================
disp("Predicciones de la red neuronal:");

// Expandir sesgos para todas las muestras
b1_exp = repmat(b1, size(X, 1), 1);
b2_exp = repmat(b2, size(X, 1), 1);

// Calcular las predicciones
Y_pred = sigmoid(sigmoid(X * W1 + b1_exp) * W2 + b2_exp);

// Mostrar las predicciones en formato one-hot (seleccionando la neurona con mayor activación)
[values, indices] = max(Y_pred, 'c');
Y_pred_onehot = zeros(size(Y_pred));
for i = 1:size(Y_pred, 1)
    Y_pred_onehot(i, indices(i)) = 1;
end

disp("Predicciones (formato one-hot):");
disp(cat(2, X, Y_pred_onehot));

// Calcular precisión
correctos = sum(indices == (Y_raw + 1)); // +1 porque Y_raw va de 0-2 pero los índices de 1-3
precision = correctos / size(X, 1) * 100;
disp("Precisión: " + string(precision) + "%");
