include("../src/filterReductionWrappers.jl") 

using Pkg

ruta_entorno = joinpath(@__DIR__, "..", "envP2")

println("Activando entorno en: $ruta_entorno")
Pkg.activate(ruta_entorno)
Pkg.instantiate()

using Test
using Statistics
using MLJBase
using DataFrames

using .CustomWrappers: PearsonSelector, SpearmanSelector, KendallSelector, ANOVASelector, MutualInfoSelector

# -------------------------------------------------------------------------
# 2. SUITE DE TESTS
# -------------------------------------------------------------------------

@testset "Pruebas Unitarias PearsonSelector" begin

    # DATOS SINTÉTICOS DE PRUEBA
    # y = 1, 2, 3, 4, 5 (Lineal simple)
    y = [1.0, 2.0, 3.0, 4.0, 5.0] 

    # Feature 1: Correlación Perfecta con y (1.0)
    f1 = [1.0, 2.0, 3.0, 4.0, 5.0] 
    
    # Feature 2: Casi perfecta con y (aprox 0.99), MUY redundante con f1
    f2 = [1.1, 1.9, 3.1, 3.9, 5.1] 
    
    # Feature 3: Correlación Negativa Fuerte con y (-1.0)
    # Debería conservarse igual que f1 porque usamos abs()
    f3 = [-1.0, -2.0, -3.0, -4.0, -5.0]
    
    # Feature 4: Ruido (Correlación baja con y)
    f4 = [5.0, 1.0, 4.0, 2.0, 3.0] 

    # Construimos la Matriz X (filas x columnas)
    # Orden original: [f1, f2, f3, f4]
    X_matrix = hcat(f1, f2, f3, f4)
    # DataFrame para simular entrada real MLJ
    X_df = DataFrame(X_matrix, [:f1, :f2, :f3, :f4])

    # -------------------------------------------------------------------
    # TEST 1: Filtrado Básico contra el Target
    # -------------------------------------------------------------------
    @testset "Pruebas Unitarias PearsonSelector" begin

    # --- DATOS SINTÉTICOS ---
    y = [1.0, 2.0, 3.0, 4.0, 5.0] 
    f1 = [1.0, 2.0, 3.0, 4.0, 5.0]        # Corr 1.0
    f2 = [1.1, 1.9, 3.1, 3.9, 5.1]        # Corr ~0.99
    f3 = [-1.0, -2.0, -3.0, -4.0, -5.0]   # Corr -1.0 (Abs 1.0) -> Redundante con f1
    f4 = [5.0, 1.0, 4.0, 2.0, 3.0]        # Ruido

    X_matrix = hcat(f1, f2, f3, f4)
    X_df = DataFrame(X_matrix, [:f1, :f2, :f3, :f4])

    end

    # -------------------------------------------------------------------
    # TEST 1: Filtrado Básico contra el Target
    # -------------------------------------------------------------------
    @testset "Filtro 1: Umbral Target" begin
        # CAMBIO: feature_threshold = 1.1 (Imposible)
        # Objetivo: Desactivar el filtro de redundancia para probar SOLO el target.
        # Así f1 y f3 (que son iguales) sobreviven ambas.
        selector = PearsonSelector(0.8, 1.1) 
        
        kept_idx, _, _ = MLJBase.fit(selector, 0, X_df, y)
        
        # Ahora f3 debe sobrevivir porque hemos "apagado" la redundancia
        @test 4 ∉ kept_idx # f4 eliminado (ruido)
        @test 1 ∈ kept_idx
        @test 2 ∈ kept_idx
        @test 3 ∈ kept_idx 
        
        println("✅ Test 1 Corregido: Filtra ruido correctamente.")
    end

    # -------------------------------------------------------------------
    # TEST 2: Lógica de Prioridad en Redundancia
    # -------------------------------------------------------------------
    @testset "Filtro 2: Redundancia Inteligente" begin
        # Aquí sí ponemos umbral estricto (0.95)
        selector = PearsonSelector(0.8, 0.95) 
        
        kept_idx, _, _ = MLJBase.fit(selector, 0, X_df, y)
        
        # f1 (corr 1.0) y f2 (corr 0.99) son redundantes. 
        # f1 es "mejor" contra el target. f2 debería morir.
        @test 1 ∈ kept_idx
        @test 2 ∉ kept_idx
        
        # f1 y f3 son redundantes (corr 1.0). Una debe morir.
        # Como tienen igual score, el algoritmo borrará la segunda que encuentre (f3).
        @test 3 ∉ kept_idx 
        
        println("✅ Test 2 Pasado: Elimina redundantes priorizando target.")
    end

    # -------------------------------------------------------------------
    # TEST 3: Caso Borde (Todo Ruido)
    # -------------------------------------------------------------------
    @testset "Caso Borde: Nada cumple" begin
        # CAMBIO: target_threshold = 1.1
        # Como nuestros datos tienen correlación 1.0, necesitamos un umbral > 1.0 para que fallen.
        selector = PearsonSelector(1.1, 1.0) 
        kept_idx, _, _ = MLJBase.fit(selector, 0, X_df, y)
        
        @test isempty(kept_idx)
        
        println("✅ Test 3 Corregido: Devuelve vacío correctamente.")
    end
    
    # -------------------------------------------------------------------
    # TEST 4: Verificación de Transformación
    # -------------------------------------------------------------------
    @testset "Transformación Correcta" begin
        # CAMBIO: feature_threshold = 1.1
        # Queremos ver que devuelve las columnas f1 y f3. 
        # Para eso, debemos permitir que coexistan (desactivar redundancia).
        selector = PearsonSelector(0.8, 1.1)
        fitresult, _, _ = MLJBase.fit(selector, 0, X_df, y)
        
        X_new = MLJBase.transform(selector, fitresult, X_df)
        
        # Ahora sí esperamos f1, f2 y f3 (f4 eliminado por target)
        expected_names = ["f1", "f2", "f3"]
        
        @test issetequal(names(X_new), expected_names)
        
        println("✅ Test 4 Corregido: Nombres de columnas coinciden.")
    end

end


@testset "Integridad de Datos y Ordenamiento (Original)" begin
    y_local = [1.0, 2.0, 3.0, 4.0]
    
    # Indices originales: 1=mid, 2=noise, 3=best
    f_mid   = [1.1, 2.1, 2.9, 4.1] 
    f_noise = [5.0, 1.0, 4.0, 2.0] 
    f_best  = [1.01, 1.99, 3.01, 3.99] 
    
    df_in = DataFrame(mid=f_mid, noise=f_noise, best=f_best)
    
    selector = PearsonSelector(0.8, 1.0)
    
    fitresult, _, _ = MLJBase.fit(selector, 0, df_in, y_local)
    df_out = MLJBase.transform(selector, fitresult, df_in)
    
    # --- VERIFICACIÓN ---
    
    # 1. Construimos la Matriz Esperada
    # Como hemos configurado el wrapper para devolver el ORDEN ORIGINAL,
    # esperamos que aparezca primero 'mid' (col 1) y luego 'best' (col 3).
    # Aunque 'best' tenga mejor correlación, respetamos su posición física.
    matrix_expected = hcat(f_mid, f_best)
    
    # 2. Comprobación de valores
    @test Matrix(df_out) ≈ matrix_expected
    
    # 3. Comprobación de nombres (Orden Original: mid antes que best)
    @test names(df_out) == ["mid", "best"]
    
    println("✅ Test 5 Pasado: Se respeta el orden original de las columnas.")
end

@testset "Spearman Monotónico" begin
    # Caso: X crece, Y crece al cuadrado (No lineal, pero monótono)
    # Pearson daría < 1.0. Spearman debe dar 1.0 exacto.
    
    y = [1, 2, 3, 4, 5]
    f_mono = [1, 4, 9, 16, 25] # Relación perfecta de rangos
    f_noise = [5, 1, 4, 2, 3]  # Ruido
    
    df = DataFrame(mono=f_mono, noise=f_noise)
    
    # Umbral alto: Solo debe pasar la relación perfecta
    selector = SpearmanSelector(0.99)
    
    fitresult, _, _ = MLJBase.fit(selector, 0, df, y)
    df_out = MLJBase.transform(selector, fitresult, df)
    
    @test names(df_out) == ["mono"]
    println("✅ Spearman detectó correctamente la relación no lineal.")
end

@testset "KendallSelector: Concordancia y Discordancia" begin
    
    # DATOS DE PRUEBA
    # y: Target ordenado (1..5)
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Feature 1: Concordancia Perfecta (Sube siempre que y sube)
    # Aunque los saltos no sean lineales, el orden se mantiene -> Kendall = 1.0
    f_direct = [10.0, 12.0, 50.0, 55.0, 100.0]
    
    # Feature 2: Discordancia Perfecta (Baja siempre que y sube)
    # Kendall = -1.0 -> abs() -> 1.0. Debe sobrevivir.
    f_inverse = [5.0, 4.0, 3.0, 2.0, 1.0]
    
    # Feature 3: Ruido (Mezcla de subidas y bajadas respecto a y)
    # Kendall será bajo.
    f_noise = [2.0, 5.0, 1.0, 4.0, 3.0]
    
    # Creamos DataFrame. Orden: [direct, inverse, noise]
    df = DataFrame(direct=f_direct, inverse=f_inverse, noise=f_noise)
    
    # CONFIGURACIÓN
    # Umbral 0.8: Solo deben pasar las relaciones fuertes (direct e inverse)
    selector = KendallSelector(0.8)
    
    # EJECUCIÓN
    fitresult, cache, report = MLJBase.fit(selector, 0, df, y)
    df_out = MLJBase.transform(selector, fitresult, df)
    
    # VERIFICACIONES
    
    # 1. Verificar Scores en el reporte
    println("Scores calculados: $(report.scores)")
    @test report.scores[1] ≈ 1.0  # direct
    @test report.scores[2] ≈ 1.0  # inverse (gracias al abs)
    @test report.scores[3] < 0.5  # noise
    
    # 2. Verificar Selección de Columnas
    # Esperamos que 'noise' haya sido eliminada
    expected_names = ["direct", "inverse"]
    @test names(df_out) == expected_names
    
    # 3. Verificar Integridad de Datos (Matriz igual a las columnas originales)
    @test df_out.direct == f_direct
    @test df_out.inverse == f_inverse
    
    println("✅ Test Kendall Pasado: Detecta correctamente el orden (directo e inverso) y filtra el ruido.")
end

@testset "ANOVASelector: Significancia y Orden" begin

    # --- 1. PREPARACIÓN DE DATOS ---
    # Escenario: Clasificación Binaria (Clase 'A' y Clase 'B')
    # 20 muestras: 10 de 'A', 10 de 'B'
    y_labels = vcat(fill("A", 10), fill("B", 10))
    y = categorical(y_labels) # Target categórico

    # A. Feature Significativa (f_sig):
    # Clase A ~ Normal(10, 1), Clase B ~ Normal(20, 1)
    # F-stat será enorme, p-value ≈ 0.0
    f_sig = vcat(randn(10) .+ 10.0, randn(10) .+ 20.0)

    # B. Feature Ruido (f_noise):
    # Clase A ~ Normal(0, 1), Clase B ~ Normal(0, 1)
    # Medias iguales, F-stat bajo, p-value alto (> 0.05)
    f_noise = randn(20)

    # C. Feature Significativa 2 (f_sig2):
    # También separa bien, la ponemos al final para probar el orden.
    f_sig2 = vcat(randn(10) .- 5.0, randn(10) .+ 5.0)

    # DataFrame Input: [f_sig, f_noise, f_sig2]
    df_in = DataFrame(f_sig=f_sig, f_noise=f_noise, f_sig2=f_sig2)

    # --- 2. CONFIGURACIÓN ---
    # Alpha = 0.05 (Estándar).
    # f_sig y f_sig2 deberían pasar. f_noise debería caer.
    selector = ANOVASelector(alpha=0.05)

    # --- 3. EJECUCIÓN ---
    fitresult, _, report = MLJBase.fit(selector, 1, df_in, y)
    df_out = MLJBase.transform(selector, fitresult, df_in)

    # --- 4. VERIFICACIONES ---

    # A. Integridad de P-Values
    println("P-values reportados: $(report.p_values)")
    
    # El p-value de f_sig (idx 1) debe ser minúsculo
    @test report.p_values[1] < 0.001 
    # El p-value de f_noise (idx 2) debe ser alto (no significativo)
    # Nota: Usamos > 0.001 para ser conservadores, pero idealmente > 0.05
    @test report.p_values[2] > 0.05 
    # El p-value de f_sig2 (idx 3) debe ser minúsculo
    @test report.p_values[3] < 0.001

    # B. Selección Correcta de Columnas
    # f_noise debe desaparecer
    @test "f_noise" ∉ names(df_out)
    @test "f_sig" ∈ names(df_out)
    @test "f_sig2" ∈ names(df_out)

    # C. Verificación de ORDEN ORIGINAL
    # La salida debe ser ["f_sig", "f_sig2"] en ese orden.
    # No importa cuál de las dos tenga el p-value más pequeño, 
    # f_sig estaba antes en el dataframe original.
    @test names(df_out) == ["f_sig", "f_sig2"]

    # D. Verificación de Matriz (Integridad de datos)
    mat_out = Matrix(df_out)
    mat_expected = hcat(f_sig, f_sig2)
    @test mat_out ≈ mat_expected

    println("✅ Test ANOVA Pasado: Filtra por p-value y respeta orden original.")
end


@testset "MutualInfoSelector: No Linealidad" begin
    
    # Target Y
    y = rand(1:3, 100) # 3 Clases aleatorias
    
    # Feature 1: Copia exacta de Y (MI debe ser máxima)
    f_copy = copy(y)
    
    # Feature 2: Ruido aleatorio (MI debe ser cercana a 0)
    f_noise = rand(100)
    
    # Feature 3: Relación cuadrática perfecta con Y (Y = X^2 aprox)
    # MI detecta esto, Pearson a veces falla si es simétrico.
    f_quad = (y .^ 2) 
    
    df = DataFrame(copy=f_copy, noise=f_noise, quad=f_quad)
    
    # Umbral bajo (0.1 bits)
    selector = MutualInfoSelector(threshold=0.1, n_bins=5)
    
    fitresult, _, report = MLJBase.fit(selector, 0, df, y)
    df_out = MLJBase.transform(selector, fitresult, df)
    
    # Verificación
    # f_noise debería tener score muy bajo
    # f_copy debería tener score alto
    
    println("Scores MI: $(report.scores)")
    
    @test report.scores[1] > 0.5  # Copy
    @test report.scores[2] < 0.2  # Noise (puede tener algo por azar, pero poco)
    
    # Selección
    @test "copy" ∈ names(df_out)
    @test "noise" ∉ names(df_out)
    
    println("✅ MI Selector detectó correctamente la información.")
end