%% Camouflage Classifier 12.3
% Training a CNN to recognize animals in camouflaged backgrounds for future
% use in transfer learning for early grade tumor detection.
% All images used for training are courtesy of:
% Anabranch Network for Camouflaged Object Segmentation
%    @article{ltnghia-CVIU2019,
%      author    = {Trung-Nghia Le and Tam V. Nguyen and Zhongliang Nie and Minh-Triet Tran and Akihiro Sugimoto,
%      journal   = {Computer Vision and Image Understanding Journal}, 
%      title     = {Anabranch Network for Camouflaged Object Segmentation}, 
%      year      = {2019}, 
%      volume    = {}, 
%      number    = {}, 
%      pages     = {-}, 
%
% In this trial, the network is trained on camo images, then transfer learned on clear images, and tested on both.
%% Building the Network

% Creating/Labeling DS Camo & Clear Imgs
CamoTrainImgs = imageDatastore('Camo Train','IncludeSubFolders',true,'LabelSource','foldernames');
CamoTestImgs = imageDatastore('Camo Test','IncludeSubFolders',true,'LabelSource','foldernames');
ClearTrainImgs = imageDatastore('Clear Train','IncludeSubFolders',true,'LabelSource','foldernames');
ClearTestImgs = imageDatastore('Clear Test','IncludeSubFolders',true,'LabelSource','foldernames');

% Image Preprocessing
CamoTrainds = augmentedImageDatastore([227 227],CamoTrainImgs,'ColorPreprocessing','gray2rgb');
CamoTestds = augmentedImageDatastore([227 227],CamoTestImgs,'ColorPreprocessing','gray2rgb');
ClearTrainds = augmentedImageDatastore([227 227],ClearTrainImgs,'ColorPreprocessing','gray2rgb');
ClearTestds = augmentedImageDatastore([227 227],ClearTestImgs,'ColorPreprocessing','gray2rgb');

% Modifying the Network
net = alexnet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(15);
layers(end) = classificationLayer;

% Setting the Training Options
trainOpts = trainingOptions('sgdm','InitialLearnRate',0.001,'LearnRateSchedule','piecewise',...
'LearnRateDropFactor',0.05,'ValidationData',CamoTrainds,'ValidationFrequency',...
10,'Shuffle','every-epoch','MaxEpochs',40,'Plots','training-progress');

% Training the network
[camo_net, info] = trainNetwork(CamoTrainds,layers,trainOpts);

% Setting tOpts 2
trainOpts2 = trainingOptions('sgdm','InitialLearnRate',0.001,'LearnRateSchedule','piecewise',...
'LearnRateDropFactor',0.05,'ValidationData',ClearTrainds,'ValidationFrequency',...
10,'Shuffle','every-epoch','MaxEpochs',50,'Plots','training-progress');

% Extracting trained Layers and modifying
layers2 = camo_net.Layers;

% Training a new network using the layers from the previously trained network
[clear_net, info] = trainNetwork(ClearTrainds,layers2,trainOpts2);

% Testing Clear Images
ClearPreds = classify(clear_net,ClearTestds);
truetest1 = ClearTestImgs.Labels;
nnz(ClearPreds == truetest1)/numel(ClearPreds)
confusionchart(truetest1,ClearPreds);
title('Clear Predictions')

% Testing Camo Images
CamoPreds = classify(clear_net,CamoTestds);
truetest2 = CamoTestImgs.Labels;
nnz(CamoPreds == truetest2)/numel(CamoPreds)
figure;
confusionchart(truetest2,CamoPreds);
title('Camo Predictions')
