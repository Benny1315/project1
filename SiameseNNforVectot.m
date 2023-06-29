close all;
clear;


data = num2cell(H2SSGoaly(:,1:250));  % Convert matrix to cell array

H2S_label = 'H2S gases';
NH3_label = 'NH3 gases';

data(1:4000, 1) = {H2S_label};
data(4001:end, 1) = {NH3_label};

% Assuming you have a cell matrix 'data' where each row represents a piece of information
% and the labels are stored in the first column
labels = categorical(data(:, 1));
vectors = cat(1, data{:, 2:end});


%% 
layers = [
    fullyConnectedLayer(64, 'WeightsInitializer', 'narrow-normal', 'BiasInitializer', 'narrow-normal')
    reluLayer
    fullyConnectedLayer(128, 'WeightsInitializer', 'narrow-normal', 'BiasInitializer', 'narrow-normal')
    reluLayer
    fullyConnectedLayer(128, 'WeightsInitializer', 'narrow-normal', 'BiasInitializer', 'narrow-normal')
    reluLayer
    fullyConnectedLayer(256, 'WeightsInitializer', 'narrow-normal', 'BiasInitializer', 'narrow-normal')
    reluLayer
    fullyConnectedLayer(4096, 'WeightsInitializer', 'narrow-normal', 'BiasInitializer', 'narrow-normal')
];

lgraph = layerGraph(layers);

% Create a sequence input layer and connect it to the first fully connected layer
inputSize = size(vectors, 2);
inputLayer = sequenceInputLayer(inputSize);
lgraph = addLayers(lgraph, inputLayer);
lgraph = connectLayers(lgraph, 'InputLayer', 'FullyConnectedLayer1');

% To train the network with a custom training loop and enable automatic differentiation,
% convert the layer graph to a dlnetwork object.
net = dlnetwork(lgraph);

% Create the weights for the final fully connected operation.
% Initialize the weights by sampling a random selection from a 
% narrow normal distribution with standard deviation of 0.01.
fcWeights = dlarray(0.01*randn(1, 4096));
fcBias = dlarray(0.01*randn(1, 1));

fcParams = struct(...
    'FcWeights', fcWeights, ...
    'FcBias', fcBias);
%% 


numIterations = 100;
miniBatchSize = 10;

learningRate = 6e-5;
gradDecay = 0.9;
gradDecaySq = 0.99;

figure
C = colororder;
lineLossTrain = animatedline('Color', C(2,:));
ylim([0 inf])
xlabel('Iteration')
ylabel('Loss')
grid on

trailingAvgSubnet = [];
trailingAvgSqSubnet = [];
trailingAvgParams = [];
trailingAvgSqParams = [];

start = tic;

% Loop over mini-batches.
for iteration = 1:numIterations

    % Extract mini-batch of data and corresponding labels
    [X, miniBatchLabels] = getMiniBatchData(vectors, labels, miniBatchSize);

    % Convert mini-batch of data to dlarray
    X = dlarray(X, 'SSCB');

    % Reshape the input data to be compatible with fully connected layers
    X = reshape(X, [1 1 inputSize miniBatchSize]);

    % Evaluate the model loss and gradients using dlfeval and the modelLoss
    % function listed at the end of the example.
    [loss, gradientsSubnet, gradientsParams] = dlfeval(@modelLoss, net, fcParams, X, miniBatchLabels);

    % Update the Siamese subnetwork parameters.
    [net, trailingAvgSubnet, trailingAvgSqSubnet] = adamupdate(net, gradientsSubnet, ...
        trailingAvgSubnet, trailingAvgSqSubnet, iteration, learningRate, gradDecay, gradDecaySq);

    % Update the fullyconnect parameters.
    [fcParams, trailingAvgParams, trailingAvgSqParams] = adamupdate(fcParams, gradientsParams, ...
        trailingAvgParams, trailingAvgSqParams, iteration, learningRate, gradDecay, gradDecaySq);

    % Update the training loss progress plot.
    D = duration(0, 0, toc(start), 'Format', 'hh:mm:ss');
    lossValue = double(gather(extractdata(loss)));
    addpoints(lineLossTrain, iteration, lossValue);
    title(['Elapsed: ' + string(D)])
    drawnow
end



function [X, miniBatchLabels] = getMiniBatchData(data, labels, miniBatchSize)
    numData = numel(labels);
    idx = randperm(numData, miniBatchSize);
    X = data(:, idx);
    miniBatchLabels = labels(idx);
end

function [loss, gradientsSubnet, gradientsParams] = modelLoss(net, fcParams, X, miniBatchLabels)
    % Pass the data through the network.
    Y = forwardSiamese(net, fcParams, X);
    
    % Calculate binary cross-entropy loss.
    miniBatchLabels = double(onehotencode(miniBatchLabels, 2));
    loss = binarycrossentropy(Y, miniBatchLabels);
    
    % Calculate gradients of the loss with respect to the network learnable
    % parameters.
    [gradientsSubnet, gradientsParams] = dlgradient(loss, net.Learnables, fcParams);
end

