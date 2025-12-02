module StatisticalTests

using Statistics
using HypothesisTests
using PrettyTables
using DataFrames
# Asumiendo que ExperimentLab ya está cargado en el entorno principal
using ..ExperimentLab: History

export get_metric_matrix, run_friedman_test, run_wilcoxon_comparison, auto_compare_models

# ==============================================================================
# 1. Preparación de Datos
# ==============================================================================

function get_metric_matrix(histories::Vector{History}, metric_name::String)
    if isempty(histories)
        error("El vector de historiales está vacío.")
    end

    # Filtrar modelos que contienen la métrica
    valid_hists = filter(h -> haskey(h.metrics, metric_name), histories)

    if isempty(valid_hists)
        error("Ningún modelo contiene la métrica: $metric_name")
    end

    n_folds = length(valid_hists[1].metrics[metric_name])
    n_models = length(valid_hists)

    data_matrix = zeros(Float64, n_folds, n_models)
    model_names = String[]

    for (i, h) in enumerate(valid_hists)
        push!(model_names, h.name)
        data_matrix[:, i] = h.metrics[metric_name]
    end

    return data_matrix, model_names
end

# ==============================================================================
# 2. Tests Estadísticos
# ==============================================================================


function run_kruskal_wallis(data_matrix::Matrix{Float64}; alpha=0.05)
    
    groups = [data_matrix[:, i] for i in 1:size(data_matrix, 2)]
    
    kw = KruskalWallisTest(groups...)
    pv = pvalue(kw)
    
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

function paired_ttest(data_matrix::Matrix{Float64}, model_names::Vector{String}; alpha=0.05)
    
    # Determinar referencia técnica (Mayor Media)
    means = vec(mean(data_matrix, dims=1))
    ref_idx = argmax(means)
    
    ref_name = model_names[ref_idx]
    ref_vec = data_matrix[:, ref_idx]
    ref_mean = mean(ref_vec)

    println("\n--- Comparativa por Pares (Paired T-Test) ---")
    println("Referencia: $ref_name (Media: $(round(ref_mean, digits=4)))")
    
    # Construcción de la tabla
    rows = []
    sorted_indices = sortperm(means, rev=true)
    
    for i in sorted_indices
        if i == ref_idx continue end 
        
        current_vec = data_matrix[:, i]
        diff = ref_mean - mean(current_vec)
        
        # Test T-Test Pareado
        # Se asume que las muestras son dependientes (mismos folds de CV)
        if current_vec == ref_vec
            pv = 1.0
            sig_str = "-"
        else
            # OneSampleTTest(x, y) es equivalente a un Paired T-Test en Julia
            tt = OneSampleTTest(ref_vec, current_vec) 
            pv = pvalue(tt)
            sig_str = pv < alpha ? "Si" : "No"
        end
        
        push!(rows, (
            Modelo = model_names[i],
            Media = round(mean(current_vec), digits=4),
            Diferencia = round(diff, digits=4),
            P_Value = round(pv, digits=5),
            Significativo = sig_str
        ))
    end

    df_res = DataFrame(rows)
    pretty_table(
        df_res;
        header = names(df_res),
        alignment = :l,
        formatters = ft_printf("%.4f", [2, 3]) # Formato para columnas numéricas
    )
end

# ==============================================================================
# 3. Orquestador
# ==============================================================================

function auto_compare_models(histories::Vector{History}, metric_name::String="Accuracy"; alpha=0.05)
    
    mat, names = get_metric_matrix(histories, metric_name)
    n_models = length(names)
    execute_paired = true

    # 1. Test Global (KW) si hay más de 2 grupos
    if n_models > 2
        execute_paired = run_kruskal_wallis(mat, alpha=alpha)
    end

    if execute_paired
        # Test Pareado (t-test)
        paired_ttest(mat, names, alpha=alpha)
    end
    
    println("\n" * "="^60)
end

end # module