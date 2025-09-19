
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




