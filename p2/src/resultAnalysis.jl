module ResultAnalysis

using Statistics
using StatsPlots
using Plots.Measures
using PrettyTables
using CategoricalArrays
using ..ExperimentLab: History

export display_cv_table, plot_cv_results, display_holdout_table

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

    # Renderizado
    pretty_table(
        data;
        header = headers,
        alignment = :l,
        crop = :none,
        formatters = ft_printf("%.4f", 2:n_cols) # Formato solo para columnas numéricas
    )
end

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
            # En Holdout, el vector de métricas solo tiene 1 elemento
            # Asumimos que existe; si no, daría error igual que la original
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

    # Renderizado
    pretty_table(
        data;
        header = headers,
        alignment = :l,
        crop = :none,
        formatters = ft_printf("%.4f", 2:n_cols) # Formato para todas las cols de métricas
    )
end

function plot_cv_results(histories::Vector{History}, metric_name::String)
    
    # Aplanado de datos para plotting
    scenarios = String[]
    values = Float64[]
    
    for h in histories
        if haskey(h.metrics, metric_name)
            fold_data = h.metrics[metric_name]
            append!(scenarios, fill(h.name, length(fold_data)))
            append!(values, fold_data)
        end
    end

    # Ordenar resultados por mediana
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
end

end # module