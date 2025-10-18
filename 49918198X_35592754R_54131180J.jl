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
using Flux.Losses

indexOutputLayer(ann::Chain) = length(ann) - (ann[end]==softmax);


function newClassCascadeNetwork(numInputs::Int, numOutputs::Int)

    if numOutputs ==1
        ann = Chain(Dense(numInputs,numOutputs,σ))
    
    # - Si numOutputs > 2 → Dense(numInputs, numOutputs, identity) |> softmax
    else
        ann = Chain(Dense(numInputs, numOutputs, identity),softmax)
    end

    return ann
end;


function addClassCascadeNeuron(previousANN::Chain; transferFunction::Function=σ)

    # Referenciar capa de salida y capas previas
    outputLayer   = previousANN[indexOutputLayer(previousANN)]
    previousLayers = previousANN[1:(indexOutputLayer(previousANN)-1)]

    # Dimensiones de la capa de salida
    numInputsOutputLayer  = size(outputLayer.weight, 2) # entradas
    numOutputsOutputLayer = size(outputLayer.weight, 1) # salidas

    # Nueva capa con una neurona oculta extra
    newLayer = SkipConnection(
        Dense(numInputsOutputLayer, 1, transferFunction),
        (mx, x) -> vcat(x, mx)   # concatena entradas originales + salida nueva
    )

    # Nueva capa de salida según el caso
    newOutputLayer = if numOutputsOutputLayer == 1
        # Clasificación binaria (una salida con σ)
        Dense(numInputsOutputLayer + 1, 1, σ)
    else
        # Clasificación multiclase (Dense + softmax)
        Chain(
            Dense(numInputsOutputLayer + 1, numOutputsOutputLayer, identity),
            softmax
        )
    end

    # Construir la nueva red
    ann = Chain(previousLayers..., newLayer, newOutputLayer)

    # Copiar pesos de la capa de salida anterior
    if newOutputLayer isa Dense
        # Copiar a la nueva Dense (última col = 0)
        ann[end].weight[:, 1:end-1] .= outputLayer.weight
        ann[end].weight[:, end] .= 0.0f0
        ann[end].bias .= outputLayer.bias
    elseif newOutputLayer isa Chain
        # Caso softmax: la Dense está en newOutputLayer[1]
        denseLayer = ann[end][1]
        denseLayer.weight[:, 1:end-1] .= outputLayer.weight
        denseLayer.weight[:, end] .= 0.0f0
        denseLayer.bias .= outputLayer.bias
    end

    return ann
end;


function trainClassANN!(
    ann::Chain,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    trainOnly2LastLayers::Bool;
    maxEpochs::Int=1000,
    minLoss::Real=0.0,
    learningRate::Real=0.001,
    minLossChange::Real=1e-7,
    lossChangeWindowSize::Int=5,
    seed::Int=1234   # NUEVO: semilla para reproducibilidad
)

    # Fijar semilla para reproducibilidad
    Random.seed!(seed)
    
    # Preparar datos de entrada y asegurarnos de que son Float32 / Bool
    trainingInputs = Float32.(trainingDataset[1])
    trainingTargets = Bool.(trainingDataset[2])
    
    # Definir función de loss
    loss(model, x, y) = (size(y,1) == 1) ? Flux.binarycrossentropy(model(x), y) : Flux.crossentropy(model(x), y)
    
    # Configurar optimizador
    opt_state = Flux.setup(Adam(learningRate), ann)
    
    # Congelar capas si es necesario
    if trainOnly2LastLayers
        Flux.freeze!(opt_state.layers[1:(indexOutputLayer(ann)-2)])
    end
    
    # Inicializar vector de losses
    trainingLosses = Float32[]
    
    # Calcular loss inicial (ciclo 0)
    push!(trainingLosses, Float32(loss(ann, trainingInputs, trainingTargets)))
    
    # Bucle de entrenamiento
    for epoch in 1:maxEpochs
        # Entrenar una época
        Flux.train!(loss, ann, [(trainingInputs, trainingTargets)], opt_state)
        
        # Calcular loss actual
        currentTrainingLoss = Float32(loss(ann, trainingInputs, trainingTargets))
        push!(trainingLosses, currentTrainingLoss)
        
        # Criterio de parada: loss mínimo alcanzado
        if currentTrainingLoss <= minLoss
            break
        end
        
        # Criterio de parada: cambio mínimo en loss
        if length(trainingLosses) >= lossChangeWindowSize
            lossWindow = trainingLosses[end-lossChangeWindowSize+1:end]
            minLossValue, maxLossValue = extrema(lossWindow)
            if (maxLossValue - minLossValue) / minLossValue <= minLossChange
                break
            end
        end
    end
    
    return trainingLosses
end




function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunction::Function=σ,
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    
    # Trasponer matrices de inputs y targets
    inputs, targets = trainingDataset
    tInputs = Float32.(inputs') # Trasponer y convertir a Float32
    tTargets = targets'
    
    # Llamar a newClassCascadeNetwork -> devuelve red sin capas ocultas
    ann = newClassCascadeNetwork(size(tInputs, 1), size(tTargets, 1))
    # Entrenar con !trainClassANN -> devuelve valores de loss del entrenamiento
    loss = trainClassANN!(ann, (tInputs, tTargets), false; maxEpochs=maxEpochs, minLoss=minLoss, 
        learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)

    println("Resultado de la funcion: ", indexOutputLayer(ann))

    for _ in 1:maxNumNeurons

        ann = addClassCascadeNeuron(ann; transferFunction) # Añadir capa oculta en cascada
        println("Resultado de la funcion: ", indexOutputLayer(ann))
        println("Longitud: ", length(ann))

        if indexOutputLayer(ann) > 2 # Si hay capas ocultas

            println("Longitud de ANN: ", indexOutputLayer(ann))
            println("Se ha entrenado la nueva capa oculta")

            # Entrenar la red
            partLoss = trainClassANN!(ann, (tInputs, tTargets), true; maxEpochs=maxEpochs, minLoss=minLoss,
                learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)

            loss = vcat(loss, partLoss[2:end]) # Concatenar vectores de loss
        
        end

        fullLoss = trainClassANN!(ann, (tInputs, tTargets), false; maxEpochs=maxEpochs, minLoss=minLoss,
            learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)

        loss = vcat(loss, fullLoss[2:end]) # Concatenar vectores de loss
        
    end

    return (ann, loss)

end;


function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunction::Function=σ,
    maxEpochs::Int=100, minLoss::Real=0.0, learningRate::Real=0.01, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)

    # Extraer inputs y targets
    inputs, targets = trainingDataset
    targets_matrix = reshape(targets, :, 1) # convert vector to matrix
    
    return trainClassCascadeANN(maxNumNeurons, (inputs, targets_matrix);
        transferFunction=transferFunction, maxEpochs=maxEpochs, minLoss=minLoss,
        learningRate=learningRate, minLossChange=minLossChange, 
        lossChangeWindowSize=lossChangeWindowSize)
end;
    

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

HopfieldNet = Array{Float32,2}

function trainHopfield(trainingSet::AbstractArray{<:Real,2})
     # Convertir a Float32
    X = Float32.(trainingSet)
    
    # Número de instancias y características
    N, D = size(X)
    
    # Calcular matriz de pesos: W = (X' * X) / N
    W = X' * X ./ N   # Ajuste: dividir entre el número de columnas
    
    # Poner diagonal a 0
    for i in 1:D
        W[i,i] = 0.0f0
    end
    
    return W

end;
function trainHopfield(trainingSet::AbstractArray{<:Bool,2})
    # Convertir {0,1} a {-1,1} y llamar al método anterior
    realSet = 2 .* Float32.(trainingSet) .- 1
    return trainHopfield(realSet)
end

function trainHopfield(trainingSetNCHW::AbstractArray{<:Bool,4})
    N, C, H, Wd = size(trainingSetNCHW)
    # Aplanar a 2D: cada imagen como fila
    reshapedSet = reshape(trainingSetNCHW, N, C*H*Wd)
    # Llamar a la versión 2D
    return trainHopfield(reshapedSet)
end;

function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    # Convertir a Float32
    x = Float32.(S)
    
    # Multiplicar por la matriz de pesos
    y = ann * x
    
    # Aplicar signo para obtener -1 o 1
    nextState = Float32.(sign.(y))
    
    return nextState
end;
function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Bool,1})
    # Convertir de {0,1} a {-1,1}
    realState = 2 .* S .- 1
    
    # Llamar al método anterior
    nextStateReal = stepHopfield(ann, realState)
    
    # Convertir de {-1,1} a Bool: >= 0 → true, < 0 → false
    nextStateBool = nextStateReal .>= 0
    
    return nextStateBool
end;


function runHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    prev_S = nothing;
    prev_prev_S = nothing;
    while S!=prev_S && S!=prev_prev_S
        prev_prev_S = prev_S;
        prev_S = S;
        S = stepHopfield(ann, S);
    end;
    return S
end;
function runHopfield(ann::HopfieldNet, dataset::AbstractArray{<:Real,2})
    outputs = copy(dataset);
    for i in 1:size(dataset,1)
        outputs[i,:] .= runHopfield(ann, view(dataset, i, :));
    end;
    return outputs;
end;
function runHopfield(ann::HopfieldNet, datasetNCHW::AbstractArray{<:Real,4})
    outputs = runHopfield(ann, reshape(datasetNCHW, size(datasetNCHW,1), size(datasetNCHW,3)*size(datasetNCHW,4)));
    return reshape(outputs, size(datasetNCHW,1), 1, size(datasetNCHW,3), size(datasetNCHW,4));
end;





function addNoise(datasetNCHW::AbstractArray{<:Bool,4}, ratioNoise::Real)
    # Copia el array para no modificar el original
    noisySet = copy(datasetNCHW)
    
    # Número total de elementos
    totalElems = length(noisySet)
    
    # Número de píxeles a modificar
    numNoise = Int(round(totalElems * ratioNoise))
    
    if numNoise > 0
        # Elegir índices aleatorios
        indices = randperm(totalElems)[1:numNoise]
        # Invertir los valores en esos índices
        noisySet[indices] .= .!noisySet[indices]
    end
    
   
    
    return noisySet
end;

function cropImages(datasetNCHW::AbstractArray{<:Bool,4}, ratioCrop::Real)
    croppedSet = copy(datasetNCHW)
    # Obtener dimensiones
    N, C, H, W = size(croppedSet)
    
    # Número de columnas a "borrar" por imagen
    numCrop = Int(round(W * ratioCrop))
    
    if numCrop > 0
        # Índices de columnas a poner a 0
        colsToCrop = (W - numCrop + 1):W
        # Poner a false (negro) esas columnas
        croppedSet[:, :, :, colsToCrop] .= false
    end
    
    @assert size(croppedSet) == size(datasetNCHW)  # Comprobación de tamaño
    return croppedSet

end;

function randomImages(numImages::Int, resolution::Int)
    
    # Generar matriz aleatoria normal (numImages x 1 x resolution x resolution)
    randomMat = randn(Float32, numImages, 1, resolution, resolution)
    
    # Convertir a boolean: valores >0 → true, <=0 → false
    boolImages = randomMat .> 0
    
    return boolImages

end;

function averageMNISTImages(imageArray::AbstractArray{<:Real,4}, labelArray::AbstractArray{Int,1})
    labels = unique(labelArray)
    N = length(labels)
    # Dimensiones de las imágenes
    _, C, H, W = size(imageArray)

     # Matriz de salida: N x C x H x W
    avgImages = zeros(eltype(imageArray), N, C, H, W)
    
    # Un único bucle: promedio de cada dígito
    for i in 1:N
        digit = labels[i]
        avgImages[i, :, :, :] .= dropdims(mean(imageArray[labelArray .== digit, :, :, :], dims=1), dims=1)
    end
    
    return (avgImages, labels)

end;

function classifyMNISTImages(imageArray::AbstractArray{<:Bool,4}, templateInputs::AbstractArray{<:Bool,4}, templateLabels::AbstractArray{Int,1})
    numImages = size(imageArray, 1)
    outputs = fill(-1, numImages)  # Inicializar a -1
    tl= length(templateLabels)
    # Un bucle sobre plantillas
    for idx in 1:tl
        template = templateInputs[[idx], :, :, :]
        # Comparación con todas las imágenes (broadcast)
        indicesCoincidence = vec(all(imageArray .== template, dims=(2,3,4)))
        outputs[indicesCoincidence] .= templateLabels[idx]
    end
    
    return outputs
end;

function calculateMNISTAccuracies(datasetFolder::String, labels::AbstractArray{Int,1}, threshold::Real)
    # Verificar que no hay etiquetas repetidas
    @assert length(labels) == length(unique(labels)) "El vector de etiquetas no debe contener valores repetidos"
    
    # 1. Cargar el dataset MNIST (usando labels como argumento keyword)
    (trainImages, trainLabels, testImages, testLabels) = loadMNISTDataset(datasetFolder; labels=labels, datasetType=Float32)
    
    # 2. Obtener las imágenes plantilla promediando las de entrenamiento
    (templateImages, templateLabels) = averageMNISTImages(trainImages, trainLabels)
    
    # 3. Umbralizar las tres matrices de imágenes
    trainImagesBinary = trainImages .>= threshold
    testImagesBinary = testImages .>= threshold
    templateImagesBinary = templateImages .>= threshold
    
    # 4. Entrenar la red de Hopfield con las plantillas umbralizadas
    hopfieldNet = trainHopfield(templateImagesBinary)
    
    # 5. Calcular precisión en el conjunto de entrenamiento
    # Ejecutar la red con las imágenes de entrenamiento
    trainReconstructions = runHopfield(hopfieldNet, trainImagesBinary)
    # Clasificar las reconstrucciones
    trainPredictions = classifyMNISTImages(trainReconstructions, templateImagesBinary, templateLabels)
    # Calcular precisión
    trainAccuracy = mean(trainPredictions .== trainLabels)
    
    # 6. Calcular precisión en el conjunto de test
    # Ejecutar la red con las imágenes de test
    testReconstructions = runHopfield(hopfieldNet, testImagesBinary)
    # Clasificar las reconstrucciones
    testPredictions = classifyMNISTImages(testReconstructions, templateImagesBinary, templateLabels)
    # Calcular precisión
    testAccuracy = mean(testPredictions .== testLabels)
    
    # 7. Devolver tupla con precisiones
    return (trainAccuracy, testAccuracy)
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------

# using ScikitLearn: @sk_import, fit!, predict
# @sk_import svm: SVC
import MLJBase: fit!
using MLJ, LIBSVM, MLJLIBSVMInterface, MLJBase
SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
import Main.predict
predict(model, inputs::AbstractArray) = collect(MLJ.predict(model, MLJ.table(inputs))) .== true


using Base.Iterators: partition
using StatsBase
using Random 

Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}


function batchInputs(batch::Batch)
    return batch[1]
end;


function batchTargets(batch::Batch)
    return batch[2]
end;


function batchLength(batch::Batch)
    return size(batchInputs(batch), 1)
end;

function selectInstances(batch::Batch, indices::Any)
    selected_inputs = batchInputs(batch)[indices, :]
    selected_targets = batchTargets(batch)[indices]
    return (selected_inputs, selected_targets)
end;

function joinBatches(batch1::Batch, batch2::Batch)
    new_inputs = vcat(batchInputs(batch1), batchInputs(batch2))
    new_targets = vcat(batchTargets(batch1), batchTargets(batch2))
    return (new_inputs, new_targets)
end;


function divideBatches(dataset::Batch, batchSize::Int; shuffleRows::Bool=false)
    n = batchLength(dataset)                      
    indices = collect(1:n)                        
    
    if shuffleRows
        Random.shuffle!(indices)                  
    end

    partitions = partition(indices, batchSize)    

    return [selectInstances(dataset, p) for p in partitions] 
end;

function trainSVM(dataset::Batch, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.,
    supportVectors::Batch=( Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)) , Array{eltype(dataset[2]),1}(undef,0) ) )
    
    # 1️ Unir los batches (vectores de soporte + dataset)
    trainingBatch = joinBatches(supportVectors, dataset)

    # 2️ Crear el modelo SVM
    model = SVMClassifier(
        kernel =
            kernel=="linear"  ? LIBSVM.Kernel.Linear :
            kernel=="rbf"     ? LIBSVM.Kernel.RadialBasis :
            kernel=="poly"    ? LIBSVM.Kernel.Polynomial :
            kernel=="sigmoid" ? LIBSVM.Kernel.Sigmoid : nothing,
        cost   = Float64(C),
        gamma  = Float64(gamma),
        degree = Int32(degree),
        coef0  = Float64(coef0)
    )

    # 3️ Entrenar el modelo
    mach = machine(model, MLJ.table(batchInputs(trainingBatch)), categorical(batchTargets(trainingBatch)))
    fit!(mach, verbosity=0)

    # 4️ Obtener índices de vectores de soporte
    indicesNewSupportVectors = sort(mach.fitresult[1].SVs.indices)

    # 5️ Separar índices
    N = batchLength(supportVectors)

    indices_from_supportVectors = [i for i in indicesNewSupportVectors if i <= N]
    indices_from_dataset = [i - N for i in indicesNewSupportVectors if i > N]

    # 6️ Crear batches de vectores de soporte 
    sv_from_supportVectors = selectInstances(supportVectors, indices_from_supportVectors)
    sv_from_dataset = selectInstances(dataset, indices_from_dataset)
    supportVectorsBatch = joinBatches(sv_from_supportVectors, sv_from_dataset)

    # 7️ Devolver modelo, batch de vectores de soporte e índices
    return (mach, supportVectorsBatch, (indices_from_supportVectors, indices_from_dataset))
end;

function trainSVM(batches::AbstractArray{<:Batch,1}, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)

    inputs, labels = (batchInputs(batches[1]), batchTargets(batches[1]))

    # Batch inicial vacío de vectores de soporte
    supportVectors = (
        Array{eltype(inputs),2}(undef, 0, size(inputs, 2)),
        Array{eltype(labels),1}(undef, 0)
    )

    mach = nothing  # variable para almacenar el modelo entrenado

    # Entrenamiento incremental: recorrer todos los batches
    for batch in batches
        mach, supportVectors, _ = trainSVM(batch, kernel, C;
                                            degree=degree,
                                            gamma=gamma,
                                            coef0=coef0,
                                            supportVectors=supportVectors)
    end

    return mach
end;





# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function initializeStreamLearningData(datasetFolder::String, windowSize::Int, batchSize::Int)
    
    dataset = loadStreamLearningDataset(datasetFolder) # Load dataset
    memory = selectInstances(dataset, 1:windowSize) # Take first windowSize instances
    # Separate the rest of the data
    _, labels = dataset
    n = size(labels, 1)
    rest = selectInstances(dataset, windowSize+1:n)
    batches = divideBatches(rest, batchSize; shuffleRows=false) # Divide the rest of the instances
    return (memory, batches)
end;

function addBatch!(memory::Batch, newBatch::Batch)
    
    inputsBatch, labelsBatch = newBatch
    inputsMemory, labelsMemory = memory

    n = size(labelsBatch, 1) # calculate the offset

    # Replace old memory
    inputsMemory[1:end-n,:] = inputsMemory[n+1:end,:] 
    labelsMemory[1:end-n] = labelsMemory[n+1:end] 

    # Write new batch into memory
    inputsMemory[end-n+1:end,:] = inputsBatch
    labelsMemory[end-n+1:end] = labelsBatch

end;

function streamLearning_SVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    
    memory, batches = initializeStreamLearningData(datasetFolder, windowSize, batchSize) # initialize memory and data
    model, _, _ = trainSVM(memory, kernel, C; degree=degree, gamma=gamma, coef0=coef0)
    accHistory = Float64[]

    for batch in batches
        y_pred = predict(model, batch[1])
        accuracy = mean(y_pred .== batch[2])
        push!(accHistory, accuracy)        
        addBatch!(memory, batch)
        model, _, _ = trainSVM(memory, kernel, C; degree=degree, gamma=gamma, coef0=coef0)
    end

    return accHistory

end;

function streamLearning_ISVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    
    firstBatch, batches = initializeStreamLearningData(datasetFolder, batchSize, batchSize) # initialize memory and data
    model, SVBatch, indices = trainSVM(firstBatch, kernel, C; degree=degree, gamma=gamma, coef0=coef0) # Train first batch
    instanceTimestamp = collect(batchSize:-1:1) # Create timestamp for all instances
    SVAges = instanceTimestamp[indices[2]]
    accHistory = Float64[]

    for batch in batches

        # Test model and calculate and register accuracy
        y_pred = predict(model, batch[1])
        accuracy = mean(y_pred .== batch[2])
        push!(accHistory, accuracy)  

        # Update support vectors timestamp
        SVAges .+= size(batch[1], 1) 

        # Select only newer support vectors
        svInputs, svLabels = selectInstances(SVBatch, SVAges.<=windowSize)
        SVAges = SVAges[SVAges .<= windowSize]
        SVBatch = (svInputs, svLabels)

        # Train new incremental SVM
        model, _, indices = trainSVM(batch, kernel, C; 
                                                    degree=degree, 
                                                    gamma=gamma, 
                                                    coef0=coef0, 
                                                    supportVectors = SVBatch) 
        # Create new batch
        oldSV = selectInstances(SVBatch, indices[1])
        newSV = selectInstances(batch, indices[2])
        SVBatch = joinBatches(oldSV, newSV)
        # Create new timestamp vector
        oldSVAges = SVAges[indices[1]]
        newSVAges = collect(size(batch[1], 1):-1:1)
        newSVAges = newSVAges[indices[2]]
        SVAges = vcat(oldSVAges, newSVAges)
    end

    return accHistory
end;

function euclideanDistances(dataset::Batch, instance::AbstractArray{<:Real,1})

    diff = instance' .- dataset[1] # Transpose vector and calculate differences
    sums = sum(diff.^2, dims=2) # Sum of squares
    euclidian_distances = sqrt.(sums) # Square root of sum of squares
    return vec(euclidian_distances)
end;

function nearestElements(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    
    distances = euclideanDistances(dataset, instance)
    indices = partialsortperm(distances, 1:k)
    return selectInstances(dataset, indices)

end;

function predictKNN(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    
    neighbours = nearestElements(dataset, instance, k)
    nLabels = batchTargets(neighbours)
    return StatsBase.mode(nLabels)

end;

function predictKNN(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int)
    
    return [predictKNN(dataset, instance, k) for instance in eachrow(instances)]

end;

function streamLearning_KNN(datasetFolder::String, windowSize::Int, batchSize::Int, k::Int)

    memory, batches = initializeStreamLearningData(datasetFolder, windowSize, batchSize) # initialize memory and data
    accHistory = Float64[]

    for batch in batches
        y_pred = predictKNN(memory, batchInputs(batch), k)
        accuracy = mean(y_pred .== batchTargets(batch))
        push!(accHistory, accuracy)        
        addBatch!(memory, batch)
    end

    return accHistory
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function predictKNN_SVM(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int, C::Real)


    # Get k nearest neighbors
    neighbors = nearestElements(dataset, instance, k)

    # Case: there is only one class
    if length(unique(batchTargets(neighbors))) == 1
        return unique(batchTargets(neighbors))[1]
    end

    # Build and train local SVM
    localSvm = SVMClassifier(
        kernel = LIBSVM.Kernel.Linear,
        cost   = Float64(C)
    )
    mach = machine(localSvm, MLJ.table(batchInputs(neighbors)), categorical(batchTargets(neighbors)))
    fit!(mach, verbosity=0)

    # Predict for given instance
    pred = predict(mach, reshape(instance, 1, :))
    TargetType = eltype(batchTargets(dataset))
    predValue = TargetType(pred[1])
    return predValue

end;

function predictKNN_SVM(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int, C::Real)

    predictions = [predictKNN_SVM(dataset, instance, k, C) for instance in eachrow(instances)]
    return predictions
end;
