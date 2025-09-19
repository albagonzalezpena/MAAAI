# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------

import FileIO.load
using DelimitedFiles
using JLD2
using Images

function fileNamesFolder(folderName::String, extension::String)
    
    extension = uppercase(extension) # obtener el formato
    fileNames = filter(f -> endswith(uppercase(f), ".$extension"), readdir(folderName)) # obtener los nombres de los archivos
    return fileNames .|> f -> f[1:end-(length(extension)+1)] # quitar la extensión y el punto del nombre de los archivos

end;



function loadDataset(datasetName::String, datasetFolder::String;
    datasetType::DataType=Float32)
    
    filePath = abspath(joinpath(datasetFolder, datasetName * ".tsv")) # Construir path
    isfile(filePath) || return nothing # Comprobar que existe
    raw = readdlm(filePath, '\t', String) # Leer archivo
    headers = raw[1, :] # Separar header

    # Buscar target
    target_idx = findfirst(==( "target"), headers)
    target_idx === nothing && return nothing

    # Separar entradas
    data = raw[2:end, :]
    inputs_str = data[:, setdiff(1:end, target_idx)]
    inputs = parse.(datasetType, inputs_str)

    # Separa y convertir targets
    targets_str = data[:, target_idx]
    targets = map(x -> Bool(parse(Int, x)), targets_str)

    return (inputs, targets)

end;


function loadImage(imageName::String, datasetFolder::String;
    datasetType::DataType=Float32, resolution::Int=128)
    
    # Construir la ruta absoluta al archivo .tif
    filePath = abspath(joinpath(datasetFolder, imageName * ".tif"))

    # Si el archivo no existe, devolver nothing
    isfile(filePath) || return nothing

    # Cargar imagen
    img = load(filePath)

    # Convertir a escala de grises y redimensionar
    img_gray = gray.(img)                
    img_resized = imresize(img_gray, (resolution, resolution))

    # Convertir a matriz del tipo especificado
    img_matrix = convert.(datasetType, img_resized)  # broadcasting sobre cada elemento

    return img_matrix
end;


function convertImagesNCHW(imageVector::Vector{<:AbstractArray{<:Real,2}})
    imagesNCHW = Array{eltype(imageVector[1]), 4}(undef, length(imageVector), 1, size(imageVector[1],1), size(imageVector[1],2));
    for numImage in Base.OneTo(length(imageVector))
        imagesNCHW[numImage,1,:,:] .= imageVector[numImage];
    end;
    return imagesNCHW;
end;


function loadImagesNCHW(datasetFolder::String;
    datasetType::DataType=Float32, resolution::Int=128)
    
    fileNames = fileNamesFolder(datasetFolder, "tif") # Obtener los nombres de los archivos
    images = loadImage.(fileNames, datasetFolder; datasetType=datasetType, resolution=resolution) # Obtener imagen en matriz
    imagesNchw = convertImagesNCHW(images) # Convertir a NCHW

    return imagesNchw
end;


showImage(image      ::AbstractArray{<:Real,2}                                      ) = display(Gray.(image));
showImage(imagesNCHW ::AbstractArray{<:Real,4}                                      ) = display(Gray.(     hcat([imagesNCHW[ i,1,:,:] for i in 1:size(imagesNCHW ,1)]...)));
showImage(imagesNCHW1::AbstractArray{<:Real,4}, imagesNCHW2::AbstractArray{<:Real,4}) = display(Gray.(vcat(hcat([imagesNCHW1[i,1,:,:] for i in 1:size(imagesNCHW1,1)]...), hcat([imagesNCHW2[i,1,:,:] for i in 1:size(imagesNCHW2,1)]...))));



function loadMNISTDataset(datasetFolder::String; labels::AbstractArray{Int,1}=0:9, datasetType::DataType=Float32)
    
    # Cargar dataset de la carpeta
    filePath = abspath(joinpath(datasetFolder, "MNIST.jld2"))
    if !isfile(filePath)
        return nothing
    end

    # Cargar datos
    file = jldopen(filePath, "r")
    trainX = file["train_imgs"]
    trainY = file["train_labels"]
    testX = file["test_imgs"]
    testY = file["test_labels"]
    close(file)

    println("TrainX original size: ", size(trainX[1]))
    println("X original size: ", size(testX[1]))

    # Convertir entradas al tipo deseado
    trainX = convert.(Matrix{datasetType}, trainX)
    testX  = convert.(Matrix{datasetType}, testX)

    println("Conversion made")

    # Convertir etiquetas a enteros (por si vinieran en otro tipo)
    trainY = Int.(trainY)
    testY  = Int.(testY)

    # Escoger los datos deseados
    if -1 in labels
        # Fijar los labels a -1 (agrupar en uno o varios contra todos)
        allowed = setdiff(labels, -1)
        trainY[.!in.(trainY, [allowed])] .= -1
        testY[.!in.(testY, [allowed])]   .= -1
    else
        # Filtrar solo las etiquetas pedidas
        trainIdx = in.(trainY, [labels])
        testIdx  = in.(testY, [labels])

        trainX = trainX[trainIdx]
        trainY = trainY[trainIdx]

        testX = testX[testIdx]
        testY = testY[testIdx]
    end

    # Convertir a imagenes NCHW
    trainXNchw = convertImagesNCHW(trainX)
    testXNchw = convertImagesNCHW(testX)

    return (trainXNchw, trainY, testXNchw, testY)

end;


function intervalDiscreteVector(data::AbstractArray{<:Real,1})
    # Ordenar los datos
    uniqueData = sort(unique(data));
    # Obtener diferencias entre elementos consecutivos
    differences = sort(diff(uniqueData));
    # Tomar la diferencia menor
    minDifference = differences[1];
    # Si todas las diferencias son multiplos exactos (valores enteros) de esa diferencia, entonces es un vector de valores discretos
    isInteger(x::Float64, tol::Float64) = abs(round(x)-x) < tol
    return all(isInteger.(differences./minDifference, 1e-3)) ? minDifference : 0.
end;



function cyclicalEncoding(data::AbstractArray{<:Real,1})

    m = intervalDiscreteVector(data) # Calcular m
    minVal = minimum(data)
    maxVal = maximum(data)

    # Evitar división por cero
    denom = maxVal - minVal + m
    if denom == 0
        sines = zeros(length(data))
        cosines = ones(length(data))
        return (sines, cosines)
    end

    # Calcular ángulos
    angles = 2π .* (data .- minVal) ./ denom

    # Senos y cosenos
    sines = sin.(angles)
    cosines = cos.(angles)

    return (sines, cosines)

end;



function loadStreamLearningDataset(datasetFolder::String; datasetType::DataType=Float32)
    
    # Crear paths
    inputPath = abspath(joinpath(datasetFolder, "elec2_data.dat"))
    labelsPath = abspath(joinpath(datasetFolder, "elec2_label.dat"))

    # Leer archivos
    inputs = readdlm(inputPath)
    labels = readdlm(labelsPath)

    boolLabels = vec(Bool.(labels)) # Convertir a bool y vectorizar

    # Preprocesado de los indices
    cleanIdx = setdiff(1:size(inputs, 2), [1, 4]) # Eliminar columnas 1 y 4 de los inputs
    cleanInputs = inputs[:, cleanIdx]
    sin, cos = cyclicalEncoding(cleanInputs[:, 1]) # Hacer cyclical encoding de la primera columna
    prepInputs = hcat(sin, cos, cleanInputs[:, 2:end]) # Concatenar dataset

    # Convertir al tipo de dato deseado
    prepInputs = convert(Matrix{datasetType}, prepInputs)

    return (prepInputs, boolLabels)

end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Flux

indexOutputLayer(ann::Chain) = length(ann) - (ann[end]==softmax);

function newClassCascadeNetwork(numInputs::Int, numOutputs::Int)

    if numOutputs ==2
        ann = Chain(Dense(numInputs,numOutputs,σ))
    
    # - Si numOutputs > 2 → Dense(numInputs, numOutputs, identity) |> softmax
    else
        ann = Chain(Dense(numInputs, numOutputs, identity), softmax)
    end

    return ann
end;


function addClassCascadeNeuron(previousANN::Chain; transferFunction::Function=σ)
    # 1️⃣ Obtener capa de salida y capas previas
    outputLayer = previousANN[indexOutputLayer(previousANN)]
    previousLayers = previousANN[1:(indexOutputLayer(previousANN)-1)]

    # 2️⃣ Número de entradas y salidas de la capa de salida
    numInputsOutputLayer = size(outputLayer.weight, 2)
    numOutputsOutputLayer = size(outputLayer.weight, 1)

    # 3️⃣ Crear nueva capa con SkipConnection
    nuevaCapa = SkipConnection(
        Dense(numInputsOutputLayer, 1, transferFunction),
        (mx, x) -> vcat(x, mx)
    )

    # 4️⃣ Crear nueva RNA
    if typeof(outputLayer) <: Dense && outputLayer.σ === σ && numOutputsOutputLayer == 2
        # 2 clases → sigmoide
        nuevaSalida = Dense(numInputsOutputLayer + 1, numOutputsOutputLayer, σ)
    else
        # más de 2 clases → identidad + softmax
        nuevaSalida = Chain(
            Dense(numInputsOutputLayer + 1, numOutputsOutputLayer, identity),
            softmax
        )
    end

    ann = Chain(previousLayers..., nuevaCapa, nuevaSalida)

    # 5️⃣ Copiar pesos de la capa de salida de previousANN
    if typeof(nuevaSalida) <: Dense
        # matriz de pesos: última columna a 0, resto igual
        ann[end].weight[:, 1:end-1] .= outputLayer.weight
        ann[end].weight[:, end] .= 0.0f0
        # bias igual
        ann[end].bias .= outputLayer.bias
    elseif typeof(nuevaSalida) <: Chain
        # en caso de softmax (Dense + softmax)
        denseLayer = nuevaSalida[1]
        denseLayer.weight[:, 1:end-1] .= outputLayer.weight
        denseLayer.weight[:, end] .= 0.0f0
        denseLayer.bias .= outputLayer.bias
    end

    return ann
end;

using Flux

function trainClassANN!(
    ann::Chain,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    trainOnly2LastLayers::Bool;
    maxEpochs::Int=1000,
    minLoss::Real=0.0,
    learningRate::Real=0.001,
    minLossChange::Real=1e-7,
    lossChangeWindowSize::Int=5
)
    X, Y = trainingDataset
    X = Float32.(X)  # asegurarse de que las entradas son Float32
    Y = Float32.(Y)  # también las salidas

    # Función de pérdida: cross-entropy (como en FAA)
    loss(x, y) = Flux.logitcrossentropy(ann(x), y)

    # Optimizer
    opt_state = Flux.setup(Adam(learningRate), ann)

    # Congelar capas si se entrenan solo las 2 últimas
    if trainOnly2LastLayers && length(ann) > 1
        Flux.freeze!(opt_state.layers[1:(indexOutputLayer(ann)-2)])
    end

    trainingLosses = Float32[]  # vector de losses
    push!(trainingLosses, loss(X, Y))  # ciclo 0

    for epoch in 1:maxEpochs
        # paso de entrenamiento
        Flux.train!(loss, ann, [(X, Y)], opt_state)

        # calcular y guardar loss
        currentLoss = loss(X, Y)
        push!(trainingLosses, currentLoss)

        # criterio: loss mínimo
        if currentLoss <= minLoss
            break
        end

        # criterio: cambio mínimo de loss en ventana
        if length(trainingLosses) >= lossChangeWindowSize
            lossWindow = trainingLosses[end-lossChangeWindowSize+1:end]
            minLossValue, maxLossValue = extrema(lossWindow)
            if (maxLossValue - minLossValue)/minLossValue <= minLossChange
                break
            end
        end
    end

    return trainingLosses
end



using Flux

function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
    transferFunction::Function=σ,
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001,
    minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)

    # --- Preparar datos: transponer y convertir entradas a Float32 ---
    X, Y = trainingDataset
    X = Float32.(X')  # instancias en columnas
    Y = Y'

    # --- Crear RNA inicial sin capas ocultas ---
    numInputs = size(X, 1)
    numOutputs = size(Y, 1)
    ann = newClassCascadeNetwork(numInputs, numOutputs)

    # --- Entrenar RNA inicial ---
    globalLosses = trainClassANN!(ann, (X, Y), false;
                        maxEpochs=maxEpochs, minLoss=minLoss,
                        learningRate=learningRate,
                        minLossChange=minLossChange,
                        lossChangeWindowSize=lossChangeWindowSize)

    # --- Bucle para añadir neuronas ---
    for i in 1:maxNumNeurons
        # Añadir neurona en cascada
        ann = addClassCascadeNeuron(ann; transferFunction=transferFunction)

        # Entrenar solo las dos últimas capas si la red tiene más de 1 capa
        if length(ann) > 1
            losses_last = trainClassANN!(ann, (X, Y), true;
                            maxEpochs=maxEpochs, minLoss=minLoss,
                            learningRate=learningRate,
                            minLossChange=minLossChange,
                            lossChangeWindowSize=lossChangeWindowSize)
            # Concatenar, evitando duplicar el primer valor
            globalLosses = vcat(globalLosses, losses_last[2:end])
        end

        # Entrenar toda la red
        losses_full = trainClassANN!(ann, (X, Y), false;
                        maxEpochs=maxEpochs, minLoss=minLoss,
                        learningRate=learningRate,
                        minLossChange=minLossChange,
                        lossChangeWindowSize=lossChangeWindowSize)
        globalLosses = vcat(globalLosses, losses_full[2:end])
    end

    return (ann, globalLosses)
end


function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
    transferFunction::Function=σ,
    maxEpochs::Int=100, minLoss::Real=0.0, learningRate::Real=0.01,
    minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)

    X, y_vector = trainingDataset

    # Convertir el vector de salidas en matriz (una fila por salida)
    Y = reshape(y_vector, 1, :)

    # Llamar a la función anterior que maneja matrices de salida
    return trainClassCascadeANN(maxNumNeurons, (X, Y);
        transferFunction=transferFunction,
        maxEpochs=maxEpochs, minLoss=minLoss,
        learningRate=learningRate,
        minLossChange=minLossChange,
        lossChangeWindowSize=lossChangeWindowSize)
end




