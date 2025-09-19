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
    numInputsOutputLayer  = size(outputLayer.weight, 2)
    numOutputsOutputLayer = size(outputLayer.weight, 1)

    # Nueva capa con una neurona oculta extra
    nuevaCapa = SkipConnection(
        Dense(numInputsOutputLayer, 1, transferFunction),
        (mx, x) -> vcat(x, mx)   # concatena entradas originales + salida nueva
    )

    # Nueva capa de salida según el caso
    nuevaSalida = if outputLayer isa Dense && outputLayer.σ === σ && numOutputsOutputLayer == 1
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
    ann = Chain(previousLayers..., nuevaCapa, nuevaSalida)

    # Copiar pesos de la capa de salida anterior
    if nuevaSalida isa Dense
        # Copiar a la nueva Dense (última col = 0)
        ann[end].weight[:, 1:end-1] .= outputLayer.weight
        ann[end].weight[:, end] .= 0.0f0
        ann[end].bias .= outputLayer.bias
    elseif nuevaSalida isa Chain
        # Caso softmax: la Dense está en nuevaSalida[1]
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
    lossChangeWindowSize::Int=5
)

    # Datos de entrenamiento
    X, Y = trainingDataset

    # Definir función de loss (igual que en FAA → binary crossentropy / log loss)
    loss_fn(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)

    # Definir optimizador
    opt_state = Flux.setup(Adam(learningRate), ann)

    # Congelar capas si corresponde
    if trainOnly2LastLayers
        Flux.freeze!(opt_state.layers[1:(indexOutputLayer(ann)-2)])
    end

    # Vector de pérdidas
    trainingLosses = Float32[]

    # Calcular loss inicial (ciclo 0)
    push!(trainingLosses, loss_fn(ann(X), Y) |> Float32)

    # Entrenamiento
    for epoch in 1:maxEpochs
        # Paso de entrenamiento
        grads = Flux.gradient(ann) do model
            loss_fn(model(X), Y)
        end
        Flux.update!(opt_state, ann, grads)

        # Calcular y guardar loss
        current_loss = loss_fn(ann(X), Y) |> Float32
        push!(trainingLosses, current_loss)

        # ---- Criterios de parada ----
        # 1) Si el loss ya es suficientemente bajo
        if current_loss <= minLoss
            println("Parada temprana: loss <= minLoss en epoch $epoch")
            break
        end

        # 2) Si el cambio relativo en la ventana es demasiado pequeño
        if length(trainingLosses) >= lossChangeWindowSize
            lossWindow = trainingLosses[end-lossChangeWindowSize+1:end]
            minLossValue, maxLossValue = extrema(lossWindow)
            if (maxLossValue - minLossValue) / minLossValue <= minLossChange
                println("Parada temprana: cambio en ventana <= minLossChange en epoch $epoch")
                break
            end
        end
    end

    return trainingLosses
end;


function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunction::Function=σ,
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    
    # Trasponer matrices de inputs y targets
    inputs, targets = trainingDataset
    tInputs = Float32.(inputs') # Trasponer y convertir a Float32
    tTargets = targets'
    
    # Llamar a newClassCascadeNetwork -> devuelve red sin capas ocultas
    ann = newClassCascadeNetwork(numInputs=size(tInputs, 1), numOutputs=size(tTargets, 1))
    # Entrenar con !trainClassANN -> devuelve valores de loss del entrenamiento
    loss = trainClassANN!(ann, (tInputs, tTargets), false; maxEpochs=maxEpochs, minLoss=minLoss, 
        learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)

    for i in 1:maxNumNeurons

        # Llamar a addClassCascadeNeuron para añadir neurona
        if length(dense_layers) > 1 # Si hay capas ocultas

            # Entrenar la red
            newLosses = trainClassANN!(ann, (tInputs, tTargets), true; maxEpochs=maxEpochs, minLoss=minLoss,
                learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)

            loss = vcat(loss, newLosses[2:end]) # Concatenar vectore de loss
        
        end

        newLosses = trainClassANN!(ann, (tInputs, tTargets), false; maxEpochs=maxEpochs, minLoss=minLoss,
            learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)

        loss = vcat(loss, newLosses[2:end]) # Concatenar vectore de loss
        
    end

    return (ann, loss)

end;

function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunction::Function=σ,
    maxEpochs::Int=100, minLoss::Real=0.0, learningRate::Real=0.01, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    #
    # Codigo a desarrollar
    #
end;
    

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

HopfieldNet = Array{Float32,2}

function trainHopfield(trainingSet::AbstractArray{<:Real,2})
    #
    # Codigo a desarrollar
    #
end;
function trainHopfield(trainingSet::AbstractArray{<:Bool,2})
    #
    # Codigo a desarrollar
    #
end;
function trainHopfield(trainingSetNCHW::AbstractArray{<:Bool,4})
    #
    # Codigo a desarrollar
    #
end;

function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    #
    # Codigo a desarrollar
    #
end;
function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Bool,1})
    #
    # Codigo a desarrollar
    #
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
    #
    # Codigo a desarrollar
    #
end;

function cropImages(datasetNCHW::AbstractArray{<:Bool,4}, ratioCrop::Real)
    #
    # Codigo a desarrollar
    #
end;

function randomImages(numImages::Int, resolution::Int)
    #
    # Codigo a desarrollar
    #
end;

function averageMNISTImages(imageArray::AbstractArray{<:Real,4}, labelArray::AbstractArray{Int,1})
    #
    # Codigo a desarrollar
    #
end;

function classifyMNISTImages(imageArray::AbstractArray{<:Bool,4}, templateInputs::AbstractArray{<:Bool,4}, templateLabels::AbstractArray{Int,1})
    #
    # Codigo a desarrollar
    #
end;

function calculateMNISTAccuracies(datasetFolder::String, labels::AbstractArray{Int,1}, threshold::Real)
    #
    # Codigo a desarrollar
    #
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------

# using ScikitLearn: @sk_import, fit!, predict
# @sk_import svm: SVC

using MLJ, LIBSVM, MLJLIBSVMInterface
SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
import Main.predict
predict(model, inputs::AbstractArray) = MLJ.predict(model, MLJ.table(inputs));



using Base.Iterators
using StatsBase

Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}


function batchInputs(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function batchTargets(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function batchLength(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function selectInstances(batch::Batch, indices::Any)
    #
    # Codigo a desarrollar
    #
end;

function joinBatches(batch1::Batch, batch2::Batch)
    #
    # Codigo a desarrollar
    #
end;


function divideBatches(dataset::Batch, batchSize::Int; shuffleRows::Bool=false)
    #
    # Codigo a desarrollar
    #
end;

function trainSVM(dataset::Batch, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.,
    supportVectors::Batch=( Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)) , Array{eltype(dataset[2]),1}(undef,0) ) )
    #
    # Codigo a desarrollar
    #
end;

function trainSVM(batches::AbstractArray{<:Batch,1}, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;





# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function initializeStreamLearningData(datasetFolder::String, windowSize::Int, batchSize::Int)
    #
    # Codigo a desarrollar
    #
end;

function addBatch!(memory::Batch, newBatch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_SVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_ISVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;

function euclideanDistances(dataset::Batch, instance::AbstractArray{<:Real,1})
    #
    # Codigo a desarrollar
    #
end;

function nearestElements(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_KNN(datasetFolder::String, windowSize::Int, batchSize::Int, k::Int)
    #
    # Codigo a desarrollar
    #
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function predictKNN_SVM(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int, C::Real)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN_SVM(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int, C::Real)
    #
    # Codigo a desarrollar
    #
end;