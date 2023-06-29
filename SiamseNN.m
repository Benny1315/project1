%% 
% Load and Preprocess Training Data
% Load the training data as a image datastore using the imageDatastore function. 
% Specify the labels manually by extracting the labels from the file names and setting the Labels property.
dataFolderTrain = 'C:\Users\97250\Documents\MATLAB\final project\training\ImageDataTraining';

imdsTrain = imageDatastore(dataFolderTrain, ...
    IncludeSubfolders=true, ...
    LabelSource="none");

files = imdsTrain.Files;
parts = split(files,filesep);
labels = join(parts(:,(end-2):(end-1)),"-");
imdsTrain.Labels = categorical(labels);

%% 
layers = [
    imageInputLayer([105 105 1],Normalization="none")
    convolution2dLayer(10,64,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(7,128,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(4,128,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    convolution2dLayer(5,256,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    fullyConnectedLayer(4096,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")];

lgraph = layerGraph(layers);
net = dlnetwork(lgraph);
fcWeights = dlarray(0.01*randn(1,4096));
fcBias = dlarray(0.01*randn(1,1));

fcParams = struct(...
    "FcWeights",fcWeights,...
    "FcBias",fcBias);

%% 
% Specify Training Options
% Specify the options to use during training. Train for 10000 iterations.
numIterations = 1000;
miniBatchSize = 180;

% Specify the options for ADAM optimization:
% Set the learning rate to 0.00006.
% Set the gradient decay factor to 0.9 and the squared gradient decay factor to 0.99.
learningRate = 6e-5;
gradDecay = 0.9;
gradDecaySq = 0.99;

% Train Model
% Initialize the training progress plot.
figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

% Initialize the parameters for the ADAM solver.
trailingAvgSubnet = [];
trailingAvgSqSubnet = [];
trailingAvgParams = [];
trailingAvgSqParams = [];

start = tic;

% Loop over mini-batches.
for iteration = 1:numIterations

    % Extract mini-batch of image pairs and pair labels
    [X1,X2,pairLabels] = getSiameseBatch(imdsTrain,miniBatchSize);

    % Convert mini-batch of data to dlarray. Specify the dimension labels
    % "SSCB" (spatial, spatial, channel, batch) for image data
    X1 = dlarray(X1,"SSCB");
    X2 = dlarray(X2,"SSCB");

    % If training on a GPU, then convert data to gpuArray.
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        X1 = gpuArray(X1);
        X2 = gpuArray(X2);
    end

    % Evaluate the model loss and gradients using dlfeval and the modelLoss
    % function listed at the end of the example.
    [loss,gradientsSubnet,gradientsParams] = dlfeval(@modelLoss,net,fcParams,X1,X2,pairLabels);

    % Update the Siamese subnetwork parameters.
    [net,trailingAvgSubnet,trailingAvgSqSubnet] = adamupdate(net,gradientsSubnet, ...
        trailingAvgSubnet,trailingAvgSqSubnet,iteration,learningRate,gradDecay,gradDecaySq);

    % Update the fullyconnect parameters.
    [fcParams,trailingAvgParams,trailingAvgSqParams] = adamupdate(fcParams,gradientsParams, ...
        trailingAvgParams,trailingAvgSqParams,iteration,learningRate,gradDecay,gradDecaySq);

    % Update the training loss progress plot.
    D = duration(0,0,toc(start),Format="hh:mm:ss");
    lossValue = double(loss);
    addpoints(lineLossTrain,iteration,lossValue);
    title("Elapsed: " + string(D))
    drawnow
end

%%


dataFolderTest = 'C:\Users\97250\Documents\MATLAB\final project\training\ImageDataTest';
imdsTest = imageDatastore(dataFolderTest, ...
    IncludeSubfolders=true, ...
    LabelSource="none");

files = imdsTest.Files;
parts = split(files,filesep);
labels = join(parts(:,(end-2):(end-1)),"_");
imdsTest.Labels = categorical(labels);

numClasses = numel(unique(imdsTest.Labels))

%%
accuracy = zeros(1,5);
accuracyBatchSize = 150;

for i = 1:5
    % Extract mini-batch of image pairs and pair labels
    [X1,X2,pairLabelsAcc] = getSiameseBatch(imdsTest,accuracyBatchSize);

    % Convert mini-batch of data to dlarray. Specify the dimension labels
    % "SSCB" (spatial, spatial, channel, batch) for image data.
    X1 = dlarray(X1,"SSCB");
    X2 = dlarray(X2,"SSCB");

%     % If using a GPU, then convert data to gpuArray.
%     if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
%         X1 = gpuArray(X1);
%         X2 = gpuArray(X2);
%     end

    % Evaluate predictions using trained network
    Y = predictSiamese(net,fcParams,X1,X2);

    % Convert predictions to binary 0 or 1
    Y = gather(extractdata(Y));
    Y = round(Y);

    % Compute average accuracy for the minibatch   %% Compute accuracy over all minibatches
    accuracy(i) = sum(Y == pairLabelsAcc)/accuracyBatchSize;
end
accuracy 
averageAccuracy = mean(accuracy)*100


%%
Display a Test Set of Images with Predictions
To visually check if the network correctly identifies similar and dissimilar pairs, create a small batch of image pairs to test. Use the predictSiamese function to get the prediction for each test pair. Display the pair of images with the prediction, the probability score, and a label indicating whether the prediction was correct or incorrect.
testBatchSize = 1;

[XTest1,XTest2,pairLabelsTest] = getSiameseBatch(imdsTest,testBatchSize);
Convert the test batch of data to dlarray. Specify the dimension labels "SSCB" (spatial, spatial, channel, batch) for image data.
XTest1 = dlarray(XTest1,"SSCB");
XTest2 = dlarray(XTest2,"SSCB");
If using a GPU, then convert the data to gpuArray.
% if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
%     XTest1 = gpuArray(XTest1);
%     XTest2 = gpuArray(XTest2);
% end
Calculate the predicted probability.
YScore = predictSiamese(net,fcParams,XTest1,XTest2);
YScore = gather(extractdata(YScore));
Convert the predictions to binary 0 or 1.
YPred = round(YScore);
Extract the data for plotting.
XTest1 = extractdata(XTest1);
XTest2 = extractdata(XTest2);
Plot images with predicted label and predicted score.
f = figure;
tiledlayout(1,1);
f.Position(2) = f.Position(2);

predLabels = categorical(YPred,[0 1],["dissimilar" "similar"]);
targetLabels = categorical(pairLabelsTest,[0 1],["dissimilar","similar"]);

for i = 1:numel(pairLabelsTest)
    nexttile
    imshow([XTest1(:,:,:,i) XTest2(:,:,:,i)]);

    title("Target: "  + string(targetLabels(i)) + newline + ...       
        "Predicted: "  + string(predLabels(i)) + newline + ...  
        "Score: " + YScore(i))
end


%% 


% Supporting Functions
% Model Functions for Training and Prediction
% The function forwardSiamese is used during network training. 
% The function defines how the subnetworks and the fullyconnect and sigmoid operations combine to form the complete Siamese network. 
% forwardSiamese accepts the network structure and two training images and outputs a prediction about the similarity of the two images. 
% Within this example, the function forwardSiamese is introduced in the section Define Network Architecture.
function Y = forwardSiamese(net,fcParams,X1,X2)
% forwardSiamese accepts the network and pair of training images, and
% returns a prediction of the probability of the pair being similar (closer
% to 1) or dissimilar (closer to 0). Use forwardSiamese during training.

% Pass the first image through the twin subnetwork
Y1 = forward(net,X1);
Y1 = sigmoid(Y1);

% Pass the second image through the twin subnetwork
Y2 = forward(net,X2);
Y2 = sigmoid(Y2);

% Subtract the feature vectors
Y = abs(Y1 - Y2);

% Pass the result through a fullyconnect operation
Y = fullyconnect(Y,fcParams.FcWeights,fcParams.FcBias);

% Convert to probability between 0 and 1.
Y = sigmoid(Y);

end

The function predictSiamese uses the trained network to make predictions about the similarity of two images. 
% The function is similar to the function forwardSiamese, defined previously. 
% However, predictSiamese uses the predict function with the network instead of the forward function, 
% because some deep learning layers behave differently during training and prediction. 
% Within this example, the function predictSiamese is introduced in the section Evaluate the Accuracy of the Network.
function Y = predictSiamese(net,fcParams,X1,X2)
% predictSiamese accepts the network and pair of images, and returns a
% prediction of the probability of the pair being similar (closer to 1) or
% dissimilar (closer to 0). Use predictSiamese during prediction.

% Pass the first image through the twin subnetwork.
Y1 = predict(net,X1);
Y1 = sigmoid(Y1);

% Pass the second image through the twin subnetwork.
Y2 = predict(net,X2);
Y2 = sigmoid(Y2);

% Subtract the feature vectors.
Y = abs(Y1 - Y2);

% Pass result through a fullyconnect operation.
Y = fullyconnect(Y,fcParams.FcWeights,fcParams.FcBias);

% Convert to probability between 0 and 1.
Y = sigmoid(Y);

end


% Model Loss Function
The function modelLoss takes the Siamese dlnetwork object net, 
% a pair of mini-batch input data X1 and X2, and the label indicating whether they are similar or dissimilar. 
% The function returns the binary cross-entropy loss between the prediction and the ground truth and the gradients 
% of the loss with respect to the learnable parameters in the network. Within this example, 
% the function modelLoss is introduced in the section Define Model Loss Function.
function [loss,gradientsSubnet,gradientsParams] = modelLoss(net,fcParams,X1,X2,pairLabels)

% Pass the image pair through the network.
Y = forwardSiamese(net,fcParams,X1,X2);

% Calculate binary cross-entropy loss.
loss = binarycrossentropy(Y,pairLabels);

% Calculate gradients of the loss with respect to the network learnable
% parameters.
[gradientsSubnet,gradientsParams] = dlgradient(loss,net.Learnables,fcParams);

end

% Binary Cross-Entropy Loss Function
The binarycrossentropy function accepts the network prediction and the pair labels, 
% and returns the binary cross-entropy loss value.
function loss = binarycrossentropy(Y,pairLabels)

% Get precision of prediction to prevent errors due to floating point
% precision.
precision = underlyingType(Y);

% Convert values less than floating point precision to eps.
Y(Y < eps(precision)) = eps(precision);

% Convert values between 1-eps and 1 to 1-eps.
Y(Y > 1 - eps(precision)) = 1 - eps(precision);

% Calculate binary cross-entropy loss for each pair
loss = -pairLabels.*log(Y) - (1 - pairLabels).*log(1 - Y);

% Sum over all pairs in minibatch and normalize.
loss = sum(loss)/numel(pairLabels);

end

% Create Batches of Image Pairs
The following functions create randomized pairs of images that are similar or dissimilar, 
% based on their labels. Within this example, the function getSiameseBatch is introduced in 
% the section Create Pairs of Similar and Dissimilar Images.
% Get Siamese Batch Function
% The getSiameseBatch returns a randomly selected batch or paired images. 
% On average, this function produces a balanced set of similar and dissimilar pairs.
% function [X1,X2,pairLabels] = getSiameseBatch(imds,miniBatchSize)
% 
% pairLabels = zeros(1,miniBatchSize);
% imgSize = size(readimage(imds,1));
% X1 = zeros([imgSize 1 miniBatchSize],"single");
% X2 = zeros([imgSize 1 miniBatchSize],"single");
% 
% for i = 1:miniBatchSize
%     choice = rand(1);
% 
%     if choice < 0.5
%         [pairIdx1,pairIdx2,pairLabels(i)] = getSimilarPair(imds.Labels);
%     else
%         [pairIdx1,pairIdx2,pairLabels(i)] = getDissimilarPair(imds.Labels);
%     end
%     X1(:,:,:,i) = imds.readimage(pairIdx1) ;
%     X2(:,:,:,i) = imds.readimage(pairIdx2) ;
% end
% 
% end
function [X1, X2, pairLabels] = getSiameseBatch(imds, miniBatchSize)
    pairLabels = zeros(1, miniBatchSize);
    imgSize = size(readimage(imds, 1));
    X1 = zeros([imgSize(1:2) 1 miniBatchSize], "single");
    X2 = zeros([imgSize(1:2) 1 miniBatchSize], "single");

    for i = 1:miniBatchSize
        choice = rand(1);

        if choice < 0.5
            [pairIdx1, pairIdx2, pairLabels(i)] = getSimilarPair(imds.Labels);
        else
            [pairIdx1, pairIdx2, pairLabels(i)] = getDissimilarPair(imds.Labels);
        end
        img1 = imds.readimage(pairIdx1);
        img2 = imds.readimage(pairIdx2);
        
        % Reshape images to 4-D
        img1 = squeeze(img1);
        img2 = squeeze(img2);
        
        X1(:, :, 1, i) = img1(:, :, 1);
        X2(:, :, 1, i) = img2(:, :, 1);
    end
end

% Get Similar Pair Function
% The getSimilarPair function returns a random pair of indices for images that are in the same 
% class and the similar pair label equals 1.
function [pairIdx1,pairIdx2,pairLabel] = getSimilarPair(classLabel)

% Find all unique classes.
classes = unique(classLabel);

% Choose a class randomly which will be used to get a similar pair.
classChoice = randi(numel(classes));

% Find the indices of all the observations from the chosen class.
idxs = find(classLabel==classes(classChoice));

% Randomly choose two different images from the chosen class.
pairIdxChoice = randperm(numel(idxs),2);
pairIdx1 = idxs(pairIdxChoice(1));
pairIdx2 = idxs(pairIdxChoice(2));
pairLabel = 1;

end

% Get Disimilar Pair Function
% The getDissimilarPair function returns a random pair of indices for images that are in different
% classes and the dissimilar pair label equals 0.
function  [pairIdx1,pairIdx2,label] = getDissimilarPair(classLabel)

% Find all unique classes.
classes = unique(classLabel);

% Choose two different classes randomly which will be used to get a
% dissimilar pair.
classesChoice = randperm(numel(classes),2);

% Find the indices of all the observations from the first and second
% classes.
idxs1 = find(classLabel==classes(classesChoice(1)));
idxs2 = find(classLabel==classes(classesChoice(2)));

% Randomly choose one image from each class.
pairIdx1Choice = randi(numel(idxs1));
pairIdx2Choice = randi(numel(idxs2));
pairIdx1 = idxs1(pairIdx1Choice);
pairIdx2 = idxs2(pairIdx2Choice);
label = 0;

end






