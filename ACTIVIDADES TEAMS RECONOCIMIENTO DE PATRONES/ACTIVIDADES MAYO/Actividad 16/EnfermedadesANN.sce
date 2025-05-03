atomsLoad("ANN_Toolbox"); // ANN Toolbox

n_entradas = 13;  // Número de neuronas en la capa de entrada
n_ocultas = 10;   // Número de neuronas en la capa oculta
n_salidas = 4;    // Número de neuronas en la capa de salida (una por cada clase)

// La estructura se define con un vector para las funciones del toolbox.
N = [n_entradas, n_ocultas, n_salidas];

// Ruta del archivo de datos
file_path = "C:\Users\caam0\OneDrive\Escritorio\SEMESTRE 6\RECONOCIMIENTO DE PATRONES\ACTIVIDADES MAYO\Actividad 16\reprocessed.hungarian.data"; // [Asegúrate que esta ruta sea correcta en tu PC]

// Abrir el archivo de datos
data = mopen(file_path, 'r');

// Verificar si el archivo se abrió correctamente
if data == -1 then
    disp("Error: No se puede abrir el archivo.");
    quit;
else
    // Leer todas las líneas del archivo
    raw_data = mgetl(data);
    mclose(data);
end

// Procesar los datos línea por línea
n_rows = size(raw_data, 'r');
parsed_data = [];
for i = 1:n_rows
    line = strsubst(raw_data(i), ",", " ");
    parsed_data = [parsed_data; evstr(line)]; // <-- Esta línea procesa la fila
end

// Manejar valores faltantes (reemplazar -9 con 0)
parsed_data(parsed_data == -9) = 0;

// Separar características (X) y etiquetas (Y_raw)
X_raw = parsed_data(:, 1:n_entradas); // Características (raw)
Y_raw_numeric = parsed_data(:, n_entradas + 1); // Etiquetas numéricas

n_total_samples = size(X_raw, 'r'); // Número total de observaciones

disp(sprintf('Dataset cargado y procesado con %d observaciones.', n_total_samples));



// Escalado de características y codificación One-Hot de etiquetas.
X_scaled = zeros(size(X_raw));
for j = 1:n_entradas
    min_val = min(X_raw(:, j));
    max_val = max(X_raw(:, j));
    if max_val - min_val > 0 then
        X_scaled(:, j) = (X_raw(:, j) - min_val) / (max_val - min_val);
    else
        X_scaled(:, j) = X_raw(:, j); // Mantener si la característica es constante
    end
end
disp("Características de entrada escaladas (Min-Max).");

Y_onehot = zeros(n_total_samples, n_salidas);
for i = 1:n_total_samples
    class_index = Y_raw_numeric(i);
    if class_index >= 0 && class_index < n_salidas then
         Y_onehot(i, class_index + 1) = 1; // Ajuste a índice basado en 1 de Scilab
    else
       
    end
end

X_T = X_scaled'; // De (samples x features) a (features x samples)
Y_T = Y_onehot'; // De (samples x outputs) a (outputs x samples)

rand('seed', 43); // Semilla para reproducibilidad
W = ann_FF_init(N); // <-- Esta línea fallará si el toolbox no carga

learning_rate = [0.1, 0]; // Tasa de aprendizaje
epochs = 1000;           // Número de épocas

// --- Entrenar la Red Neuronal ---
disp(sprintf("Iniciando entrenamiento con ann_FF_Std_online para %d épocas...", epochs));
W_trained = ann_FF_Std_online(X_T, Y_T, N, W, learning_rate, epochs); // <-- Esta línea fallará si el toolbox no carga
disp("Entrenamiento completado."); // <-- Este mensaje no se mostrará si la línea anterior falla

disp("Realizando predicciones con ann_FF_run..."); // <-- Este mensaje no se mostrará si el entrenamiento falló
Y_pred_raw_T = ann_FF_run(X_T, N, W_trained); // <-- Esta línea fallará si el toolbox no carga o W_trained no se creó

predicted_classes_one_based = zeros(1, n_total_samples); // Inicialización si el código real falla
predicted_classes = predicted_classes_one_based - 1; // Convertir índice 1-based a etiqueta 0-based

correct_predictions = sum(predicted_classes' == Y_raw_numeric); // <-- Esta línea fallará si predicted_classes no se creó/es inválido
accuracy = correct_predictions / n_total_samples * 100; // <-- Usamos n_total_samples porque no dividimos en train/test aquí

disp("--- Resultados (Conceptuales) ---"); // <-- No se mostrará si falló antes
disp(sprintf('Precisión (en datos de entrenamiento): %.2f%%', accuracy)); // <-- No se mostrará si falló antes

// Mostrar algunos ejemplos de predicciones vs reales.
disp("--- Ejemplos de predicciones vs Reales (Conceptuales) ---"); 
disp("Predicho | Real");
disp("-----------------");
