// =============================================================================
// PARÁMETROS DE LA RED NEURONAL
// =============================================================================
n_entradas = 13;  // Número de entradas (características médicas)
n_ocultas = 10;   // Neuronas en la capa oculta
n_salidas = 4;    // Neuronas en la capa de salida (4 clases: 0, 1, 2, 3)
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
file_path = "C:\Users\caam0\OneDrive\Escritorio\SEMESTRE 6\RECONOCIMIENTO DE PATRONES\Actividad 9\tareas del readme\reprocessed.hungarian.data";

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

// Separar características (X) y etiquetas (Y_raw)
X = parsed_data(:, 1:13);  // Características (primeras 13 columnas)
Y_raw = parsed_data(:, 14); // Etiquetas (columna 14)

// Codificar las etiquetas en formato one-hot
Y = zeros(size(Y_raw, 1), n_salidas);  // Matriz de etiquetas one-hot
for i = 1:size(Y_raw, 1)
    if Y_raw(i) >= 0 & Y_raw(i) <= 3 then
        Y(i, Y_raw(i) + 1) = 1;  // Asignar 1 en la posición correspondiente
    end
end

// Mostrar datos de entrada y etiquetas
disp("Datos de entrada y etiquetas:");
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
    b1_expanded = repmat(b1, size(X, 1), 1);  // Expandir sesgos para todas las muestras
    Z1 = X * W1 + b1_expanded;               // Entrada ponderada + sesgo
    A1 = sigmoid(Z1);                        // Salida de la capa oculta

    // Capa de salida
    b2_expanded = repmat(b2, size(X, 1), 1);  // Expandir sesgos para todas las muestras
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
    dW2 = A1' * dZ2;                     // Gradiente de los pesos de la capa de salida
    db2 = sum(dZ2, 1);                   // Gradiente de los sesgos de la capa de salida

    // Gradientes de la capa oculta
    dZ1 = (dZ2 * W2') .* sigmoid_derivada(Z1);  // Derivada del error respecto a Z1
    dW1 = X' * dZ1;                             // Gradiente de los pesos de la capa oculta
    db1 = sum(dZ1, 1);                          // Gradiente de los sesgos de la capa oculta

    // =========================================
    // ACTUALIZACIÓN DE PESOS Y SESGOS
    // =========================================
    W2 = W2 + tasa_aprendizaje * dW2;  // Actualizar pesos de la capa de salida
    b2 = b2 + tasa_aprendizaje * db2;  // Actualizar sesgos de la capa de salida
    W1 = W1 + tasa_aprendizaje * dW1;  // Actualizar pesos de la capa oculta
    b1 = b1 + tasa_aprendizaje * db1;  // Actualizar sesgos de la capa oculta
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

// Mostrar las predicciones redondeadas
disp("Predicciones (redondeadas):");
disp(cat(2, X, fix(Y_pred + 0.5)));
