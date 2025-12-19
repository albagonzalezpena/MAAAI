module StatisticalTests

"""
    Este módulo contiene y abstrae la lógica de los test estadísticos con el final
    de seleccionar los mejores modelos de manera rigurosa en el notebook de manera 
    automatizada.
"""

using Statistics
using HypothesisTests
using PrettyTables
using DataFrames
using ..ExperimentLab: History

export get_metric_matrix, run_friedman_test, run_wilcoxon_comparison, auto_compare_models

# ==============================================================================
# Preparación de datos
# ==============================================================================

"""
    Esta función transforma los historiales de cada modelo en datos procesables
    por la librería HypothesisTest.

    Parámetros:
        - histories: vector de historiales de los modelos a examinar.
        - metric_name: nombre de la métrica sobre la que queremos realiszar el test.

    Devuelve:
        - data_matrix: matriz con las métricas de cada modelo
        - model_names: nombre de cada modelo que se quiere comparar.
"""

function get_metric_matrix(histories::Vector{History}, metric_name::String)
    if isempty(histories)
        error("El vector de historiales está vacío.")
    end

    # Filtrar modelos que contienen la métrica especificada
    valid_hists = filter(h -> haskey(h.metrics, metric_name), histories)

    if isempty(valid_hists)
        error("Ningún modelo contiene la métrica: $metric_name")
    end

    # Calcular las dimensiones de la matriz
    n_folds = length(valid_hists[1].metrics[metric_name])
    n_models = length(valid_hists)

    # Crear la matriz
    data_matrix = zeros(Float64, n_folds, n_models)
    model_names = String[]

    # Llenar cada columna con las métricas de cada modelo
    for (i, h) in enumerate(valid_hists)
        push!(model_names, h.name)
        data_matrix[:, i] = h.metrics[metric_name]
    end

    return data_matrix, model_names
end


# ==============================================================================
#  Tests Estadísticos
# ==============================================================================

"""
    Esta función ejecuta un test por grupos Kruskall-Wallis para determinar 
    si hay diferencias significativas en las medianas de las métricas entre
    varios modelos.

    Params:
        - data_matrix: matriz que contiene los datos de los modelos (cada modelo en cada columna).
        - alpha: valor alpha usado en el test estadístico.

    Returns:
        - bool: true si el resultado del test establece diferencias significativas.
"""
function run_kruskal_wallis(data_matrix::Matrix{Float64}; alpha=0.05)
    
    # Crear los grupos necesarios para el test
    groups = [data_matrix[:, i] for i in 1:size(data_matrix, 2)]
    
    # Crear test y obetner p-value
    kw = KruskalWallisTest(groups...)
    pv = pvalue(kw)
    
    # Mostrar resultados del test legibles para el experimento
    println("\n--- Test de Kruskal-Wallis (Global) ---")
    println("Hipótesis Nula (H0): Las medianas de los grupos son iguales.")
    println("p-value: $(round(pv, digits=5))")
    
    if pv < alpha
        println("Resultado: Se rechaza H0 (Diferencias significativas).")
        return true
    else
        println("Resultado: No se rechaza H0.")
        return false
    end
end

"""
    Esta función ejecuta un t-test pareado entre el modelo con una mayor métrica
    y el resto de modelos a comparar.

    Params:
        - data_matrix: matriz con los resultados de los modelos (cada modelo en cada columna).
        - model_names: vector con el nombre de los modelos.
        - alpha: nivel de significancia especificado por el usuario.

    Returns:
        - nothing
"""

function paired_ttest(data_matrix::Matrix{Float64}, model_names::Vector{String}; alpha=0.05)
    
    # Determinar modelo de referencia
    means = vec(mean(data_matrix, dims=1))
    ref_idx = argmax(means)
    
    ref_name = model_names[ref_idx]
    ref_vec = data_matrix[:, ref_idx]
    ref_mean = mean(ref_vec)

    # Mostrar modelo de referencia
    println("\n--- Comparativa por Pares (Paired T-Test) ---")
    println("Referencia: $ref_name (Media: $(round(ref_mean, digits=4)))")
    
    # Construcción de la tabla de resultados
    rows = []
    sorted_indices = sortperm(means, rev=true) # ordenar de mayor a menor media
    
    for i in sorted_indices
        if i == ref_idx continue end # No compara consigo mismo
        
        # Obtener diferencia entre medias
        current_vec = data_matrix[:, i]
        diff = ref_mean - mean(current_vec)
        
        # Test T-Test Pareado
        # Se asume que las muestras son dependientes (mismos folds de CV)
        if current_vec == ref_vec # En caso de tener dos modelos con los mismos resultados
            pv = 1.0
            sig_str = "-"
        else
            # Ejecutar t-test entre pares
            tt = OneSampleTTest(ref_vec, current_vec) 
            pv = pvalue(tt)
            sig_str = pv < alpha ? "Si" : "No" # Guardar nivel de significancia
        end
        
        # Añadir resultados a la tabla
        push!(rows, (
            Modelo = model_names[i],
            Media = round(mean(current_vec), digits=4),
            Diferencia = round(diff, digits=4),
            P_Value = round(pv, digits=5),
            Significativo = sig_str
        ))
    end

    # Generar tabla estructurada con los resultados para mostrar en el notebook
    df_res = DataFrame(rows)
    pretty_table(
        df_res;
        header = names(df_res),
        alignment = :l,
        formatters = ft_printf("%.4f", [2, 3]) # Formato para columnas numéricas
    )
end

# ==============================================================================
# Manager
# ==============================================================================

"""
    Esta fucnión actúa como trigger para ejecutar los test y contiene la lógica de 
    un test estadistico.

    Params:
        - histories: registro de datos del modelo generado tras el entrenamiento.
        - metric_name: métrica que queremos comparar.
        - alpha: nivel de significancia que queremos en nuestros tests.
"""
function auto_compare_models(histories::Vector{History}, metric_name::String="Accuracy"; alpha=0.05)
    
    # Obtener metriz de datos para ejecutar tests
    mat, names = get_metric_matrix(histories, metric_name)
    n_models = length(names)
    execute_paired = true

    # Tets por grupos si hay más de dos modelos
    if n_models > 2
        execute_paired = run_kruskal_wallis(mat, alpha=alpha)
    end

    # Ejecutar t-test pareado si el test grupal 
    # determina que hay diferencias significativas
    if execute_paired
        paired_ttest(mat, names, alpha=alpha)
    end
    
    println("\n" * "="^60)
end

end # module