module ModelFactory

using MLJ
using MLJFlux      
using Flux         
using LIBSVM       
using NearestNeighborModels 
using Random
using Optimisers
using MLJEnsembles
using EvoTrees
using MLJDecisionTreeInterface

# Exportamos las funciones factoría
export get_knn_model, get_svm_model, get_mlp_model, get_bagging_knn_model, get_evotree_model, get_adaboost_model, get_rf_model

const MLJ_SVC = @load ProbabilisticSVC pkg=LIBSVM verbosity=0
const MLJ_EvoTree = @load EvoTreeClassifier pkg=EvoTrees verbosity=0
const AdaBoostStump = @load AdaBoostStumpClassifier pkg=DecisionTree verbosity=0
const MLJ_RF = @load RandomForestClassifier pkg=DecisionTree verbosity=0


# ==============================================================================
# 1. BUILDER DINÁMICO PARA MLP 
# ==============================================================================
"""
DynamicBuilder: Estructura auxiliar para construir redes neuronales
con topología variable basada en un vector de enteros.
"""
mutable struct DynamicBuilder <: MLJFlux.Builder
    hidden_layers::Vector{Int}
end

function MLJFlux.build(b::DynamicBuilder, rng, n_in, n_out)
    layers = []
    input_dim = n_in
    
    # Crear capas ocultas
    for h in b.hidden_layers
        push!(layers, Dense(input_dim, h, relu))
        input_dim = h
    end
    
    # Capa de salida
    push!(layers, Dense(input_dim, n_out))
    
    return Chain(layers...)
end

# ==============================================================================
# 2. FACTORY: KNN
# ==============================================================================
"""
Devuelve un modelo KNN configurado.
Parámetros:
  - k: Número de vecinos (default: 5)
"""
function get_knn_model(k::Int)
    # Nota: NearestNeighborModels usa 'K' mayúscula
    return KNNClassifier(K=k)
end

# ==============================================================================
# 3. FACTORY: SVM
# ==============================================================================
"""
Devuelve un modelo SVM (Support Vector Machine) configurado.
Parámetros:
  - cost: Penalización por error (C) (default: 1.0)
  - kernel: Tipo de kernel (default: Radial)
"""
function get_svm_model(cost::Float64)
    return MLJ_SVC(cost=cost,  kernel=LIBSVM.Kernel.Linear)
end

# ==============================================================================
# 4. FACTORY: MLP (RED NEURONAL)
# ==============================================================================
"""
Devuelve un modelo de Red Neuronal (MLP) configurado.
Parámetros:
  - hidden_layers: Vector con el tamaño de las capas ocultas (ej: [100, 50])
  - epochs: Número de épocas de entrenamiento
  - learning_rate: Tasa de aprendizaje para el optimizador ADAM
"""
function get_mlp_model(hidden_layers::Vector{Int}; epochs::Int=50, learning_rate::Float64=0.001)
    
    builder = DynamicBuilder(hidden_layers)
    
    return NeuralNetworkClassifier(
        builder = builder,
        epochs = epochs,
        batch_size = 16,
        optimiser = Optimisers.Adam(learning_rate)
    )
end

# ==============================================================================
# 5. FACTORY: BAGGING (KNN BASE)
# ==============================================================================
"""
Devuelve un modelo de Bagging usando KNN como clasificador base.
Parámetros:
  - k: Número de vecinos para el KNN base (ej: 5).
  - n_estimators: Número de modelos en el ensemble (ej: 10, 50).
  - fraction: Porcentaje de datos a usar en cada bolsa (default: 0.8).
"""
function get_bagging_knn_model(k::Int, n_estimators::Int; fraction::Float64=0.8)
    
    # Definir modelo base
    base_model = KNNClassifier(K=k)
    
    # Crear ensemble
    return EnsembleModel(
        model = base_model,
        n = n_estimators,
        bagging_fraction = fraction
    )
end

# ==============================================================================
# 6. FACTORY: ADABOOST (Decision Stumps)
# ==============================================================================
"""
Devuelve un modelo AdaBoost usando 'Decision Stumps' (árboles de profundidad 1).
Este es el algoritmo clásico de AdaBoost.
Parámetros:
  - n_estimators: Número de iteraciones (estimadores).
"""
function get_adaboost_model(n_estimators::Int)
    return AdaBoostStump(
        n_iter = n_estimators  # Mapeamos el argumento al parámetro interno
    )
end


# ==============================================================================
# 7. FACTORY: EvoTree
# ==============================================================================

function get_evotree_model(n_estimators::Int; learning_rate::Float64=0.2)

    return MLJ_EvoTree(
        nrounds = n_estimators,
        eta = learning_rate,
        max_depth = 5,  
        nbins = 32      # Discretización para acelerar (estándar en boosting)
    )
end

# ==============================================================================
# 7. FACTORY: Random Forest
# ==============================================================================

"""
Devuelve un modelo Random Forest.
Parámetros:
  - n_trees: Número de árboles.
  - min_samples_split: Mínimo de muestras para dividir un nodo (control de overfitting).
"""
function get_rf_model(n_trees::Int; max_depth::Int=10)
    return MLJ_RF(
        n_trees = n_trees,
        max_depth = max_depth,         
        min_samples_split = 2,
        sampling_fraction = 1.0,       # Bagging de filas (80% datos por árbol)
        n_subfeatures = -1             # -1 significa sqrt(n_features) (Estándar en RF)
    )
end

end # module