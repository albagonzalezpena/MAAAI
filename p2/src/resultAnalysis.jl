module ResultAnalysis

"""
    Módulo adaptado a nuestra gestión de resultados de experimentos
    que nos permite generar gráficas y tablas ad-hoc para cada experimento de 
    manera sencilla abstrayendo su funcionamiento, con el fin de crear un notebook
    más claro centrado en el análisis de resultados.
"""

using Statistics
using StatsPlots
using Plots.Measures
using PrettyTables
using CategoricalArrays
using ..ExperimentLab: History
using Plots

export display_cv_table, plot_cv_results, display_holdout_table, display_confussion_matrix, process_feature_importance

"""
    Método que nos permite crear una tabla mostrando los resultados de
    la ejecución de experimentos con crossvalidation (muestra estadísticas
    como media y std).

    Params:
        - histories: vector de historiales de los modelos entrenados.
        - metric_names: métricas que se quieren mostrar en las tablas.

    Returns:
        nothing
"""

function display_cv_table(histories::Vector{History}, metric_names::Vector{String})
    
    # Construcción de cabeceras
    headers = String["Experiment"]
    for m in metric_names
        push!(headers, "$m (mean)", "$m (std)")
    end

    # Matriz de datos crudos
    n_rows = length(histories)
    n_cols = length(headers)
    data = Matrix{Any}(undef, n_rows, n_cols)

    for (i, h) in enumerate(histories)
        data[i, 1] = h.name
        
        # Llenado de métricas
        col_idx = 2
        for m in metric_names
            vals = h.metrics[m]
            data[i, col_idx]     = mean(vals)
            data[i, col_idx + 1] = std(vals)
            col_idx += 2
        end
    end

    # Ordenar por la media de la primera métrica (descendente)
    # La columna 2 siempre es la media de la primera métrica
    sort_idx = sortperm(Float64.(data[:, 2]), rev=true)
    data = data[sort_idx, :]

    # Contruir tabla estructurada
    pretty_table(
        data;
        header = headers,
        alignment = :l,
        crop = :none,
        formatters = ft_printf("%.4f", 2:n_cols)
    )
end

"""
    Esta función devuelve tablas ad-hoc para experimentos con holdout, 
    ya que no se calculan ni muestran estadísticas sobre las métricas.

    Params:
        - histories: vector con historiales de resultados de los modelos.
        - metric_names: métricas que se quieren mostrar.
    Returns:
        nothing.
"""

function display_holdout_table(histories::Vector{History}, metric_names::Vector{String})
    
    # Construcción de cabeceras (Solo nombres de métricas, sin Mean/Std)
    headers = String["Experiment"]
    append!(headers, metric_names)

    # Matriz de datos
    n_rows = length(histories)
    n_cols = length(headers)
    data = Matrix{Any}(undef, n_rows, n_cols)

    for (i, h) in enumerate(histories)
        data[i, 1] = h.name
        
        # Llenado de métricas
        col_idx = 2
        for m in metric_names
            # En holdout el vector de métricas solo tiene un elemento
            vals = h.metrics[m]
            
            # Tomamos el primer y único valor
            data[i, col_idx] = vals[1]
            col_idx += 1
        end
    end

    # Ordenar por la primera métrica (descendente)
    # La columna 2 corresponde a la primera métrica de la lista
    if n_rows > 0 && n_cols > 1
        sort_idx = sortperm(Float64.(data[:, 2]), rev=true)
        data = data[sort_idx, :]
    end

    # Contruir tabla estructurada
    pretty_table(
        data;
        header = headers,
        alignment = :l,
        crop = :none,
        formatters = ft_printf("%.4f", 2:n_cols)
    )
end

"""
    Este método nos permite obtener una representación gráfica de los
    retultados de un experimento. Crea un boxplot de los resultados de la 
    métrica especificada.

    Además, guarda la gráfica en la carpeta destinada a los resultados de 
    los experimentos.

    Params:
        - histories: vector con los resultados del experimento.
        - metric_name: nombre de la métrica a graficar.
        - name: nombre del experimento.
"""

function plot_cv_results(histories::Vector{History}, metric_name::String, name::String)
    
    # Aplanamos datos para plottinf
    scenarios = String[]
    values = Float64[]
    
    # Obtener métricas de los historiales
    for h in histories
        if haskey(h.metrics, metric_name)
            fold_data = h.metrics[metric_name]
            append!(scenarios, fill(h.name, length(fold_data)))
            append!(values, fold_data)
        end
    end

    # Ordenar resultados por mediana (estamos usando tests grupales no paramétricos)
    unique_scenarios = unique(scenarios)
    medians = map(unique_scenarios) do s
        idx = scenarios .== s
        median(values[idx])
    end
    
    # Ordenamos los niveles categóricos según la mediana
    sorted_levels = unique_scenarios[sortperm(medians)]
    
    # Convertimos a categórico ordenado para que StatsPlots respete el orden
    scenarios_cat = categorical(scenarios)
    levels!(scenarios_cat, sorted_levels)

    # Generación del gráfico
    p = boxplot(
        scenarios_cat,      
        values,             
        group = scenarios_cat, 
        legend = false,      
        title = "Comparativa: $metric_name",
        ylabel = metric_name,
        xlabel = "",         
        orientation = :vertical,    
        xrotation = 60,            
        bottom_margin = 12Plots.mm,
        left_margin = 8Plots.mm,   
        size = (800, 600),          
        outliers = true
    )

    # Guardar el gráfico
    output_dir = joinpath("..", "results")
    filename = "$(name).png"
    filepath = joinpath(output_dir, filename)
    savefig(p, filepath)

    return p
end

"""
    Función que crea una representación gráfica de la matriz de confusión obtenida
    durante el entrenamiento.

    Además, se guarda el resultado en la carpeta destinada a los resultados del
    experimento.

    Params:
        - matrix: matriz de confusión.
        - class_labels: nombres de las clases del problema ordenadas.
        - class_counts: conteo de clases para representación realista de los datos.
        - tag: nombre del experimento.
        - plot_size: tamaño del gráfico.

"""

function display_confussion_matrix(
    matrix::AbstractMatrix{<:Number}, 
    class_labels::AbstractVector{<:AbstractString}, 
    class_counts::Vector{Int}, 
    tag::String="";
    plot_size=(800, 600)
)

    # Obtener ruta de resultados
    output_dir = joinpath("..", "results")
    filename = "$(tag).png"
    filepath = joinpath(output_dir, filename)

    labels_str = string.(class_labels)
    n = size(matrix, 1)

    # Calcular proporción de true_positives para ajustar intensidad de color
    # ya que las clases no esán perfectamente balanceadas, el color se
    # define en función de la proporción de predicciones acertadas,
    # aunque solo se meustra el número raw de predicciones.
    normalized_matrix = matrix ./ class_counts

    # Definir coordenadas para centrar mejor el texto
    x_vals = 1:n
    y_vals = 1:n

    # Anotaciones usando coordenadas
    font_sz = n > 15 ? 7 : (n > 8 ? 9 : 11)

    # Crear anotaciones
    anns = vec([
        (
            j, i, 
            text(
                "$(Int(matrix[i,j]))", 
                font_sz, 
                :black, 
                :center 
            )
        ) 
        for i in 1:n, j in 1:n
    ])

    # Generar heatmap
    p = heatmap(
        x_vals, y_vals, normalized_matrix,
        xticks = (1:n, labels_str),
        yticks = (1:n, labels_str),
        title = "Confussion Matrix - $tag",
        xlabel = "Prediction", 
        ylabel = "Real",
        color = :blues,
        clims = (0, 1),
        colorbar = false,
        annotations = anns,
        yflip = true, 
        aspect_ratio = 1, 
        size = plot_size,        
        xrotation = 45, 
        left_margin = 8Plots.mm,
        bottom_margin = 8Plots.mm
    )

    savefig(p, filepath) # Guardar plot
    
    return p
end

"""
    Esta fucnión se ha creado para gestionar los datos calculados 
    de feature_importance de los modelos pertinentes. Ayuda a automatizar
    la creación de gráficos y estadísticas para comprender mejor la selección
    de características embedida en los modelos.

    Se está empleando la importanci relativa al resto de features normalizando
    a porcentaje con el fin de facilitar el análisis.

    Además, guarda los gráficos en el directorio destinado a almacenamiento
    de resultados.

    Params:
        - fi_pairs: pares (feature, importancia) del modelo.
        - tag: nombre del experimento (modelo).
        - top_n: n features más explicativas que queremos mostrar.
"""
function process_feature_importance(fi_pairs::AbstractVector, tag::String; top_n::Int=20)

    # Procesamiento básico y ordenación
    # Orden ascendente (necesario para el gráfico y 'last')
    fi_sorted = sort(fi_pairs, by = x -> last(x), rev = false)
    
    # Seleccionar top npara el gráfico
    fi_subset = last(fi_sorted, min(length(fi_sorted), top_n))
    
    names = string.(first.(fi_subset))
    values_raw_top = last.(fi_subset)
    
    # Total de importancia para normalizar
    total_importance = sum(last.(fi_pairs))
    
    # Lógica de acumulación
    all_values_desc = reverse(last.(fi_sorted)) 
    
    # Convertimos a porcentajes acumulados
    cumulative_pct = cumsum(all_values_desc ./ total_importance .* 100)
    
    # Función auxiliar para encontrar el índice donde se cruza el umbral
    function find_cutoff(threshold)
        idx = findfirst(x -> x >= threshold, cumulative_pct)
        return isnothing(idx) ? length(cumulative_pct) : idx
    end

    n_50 = find_cutoff(50.0)
    n_75 = find_cutoff(75.0)
    n_90 = find_cutoff(90.0)

    # Preparar estadísticas para la tabla
    values_top_pct = (values_raw_top ./ total_importance) .* 100
    accum_top = sum(values_top_pct)
    zeros_count = count(x -> last(x) == 0, fi_pairs)
    
    # Construcción de la matriz de estadísticas
    stats = [
        "Total Features"             length(fi_pairs);
        "Features muertas (0.0)"     zeros_count;
        "Info explicada por Top-$top_n" "$(round(accum_top, digits=2)) %";
        "Features para el 50%"       n_50;
        "Features para el 75%"       n_75;
        "Features para el 90%"       n_90
    ]
    
    # Crear tabla
    pretty_table(
        stats;
        header = nothing,
        alignment = [:l, :r],
        tf = tf_borderless
    )

    positions = 1:length(values_top_pct)

    # Gráfico
    p = bar(
        positions,
        values_top_pct,
        orientation = :h,           
        yticks = (positions, names),
        legend = false,
        color = :steelblue,
        alpha = 0.8,
        left_margin = 3mm,
        bottom_margin = 5mm,
        top_margin = 3mm,
        right_margin = 5mm,
        title = "$tag Feature Importance", 
        xlabel = "Importancia (%)",
        ylabel = "",
        grid = :x,
        xlims = (0, maximum(values_top_pct) * 1.05)
    )

    # Guardar plot
    out_dir = joinpath("..", "results")
    !isdir(out_dir) && mkpath(out_dir)
    
    safe_tag = replace(tag, " " => "_")
    path = joinpath(out_dir, "fi_$(safe_tag).png")
    
    savefig(p, path)
    
    return p
end

end # module