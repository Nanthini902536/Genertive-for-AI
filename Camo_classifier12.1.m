%% Camouflage Classifier 12.1
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
% In this trial, a network is trained on camo/clear and tested on camo and clear separately.
%% Building the Network

% Creating/Labeling DS for training & testing
CCImgs = imageDatastore('Camo+Clear Train','IncludeSubFolders',true,'LabelSource','foldernames');
CamotestImgs = imageDatastore('Camo Test','IncludeSubFolders',true,'LabelSource','foldernames');
CleartestImgs = imageDatastore('Clear Test','IncludeSubFolders',true,'LabelSource','foldernames');

% Image Preprocessing
CCtrainds = augmentedImageDatastore([227 227],CCImgs,'ColorPreprocessing','gray2rgb');
Camotestds = augmentedImageDatastore([227 227],CamotestImgs,'ColorPreprocessing','gray2rgb');
Cleartestds = augmentedImageDatastore([227 227],CleartestImgs,'ColorPreprocessing','gray2rgb');

% Modifying the Network
net = alexnet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(15);
layers(end) = classificationLayer;

% Setting the Training Options
trainOpts = trainingOptions('sgdm','InitialLearnRate',0.001,'LearnRateSchedule','piecewise',...
'LearnRateDropFactor',0.05,'ValidationData',CCtrainds,'ValidationFrequency',10,'Shuffle',...
'every-epoch','MaxEpochs',40,'Plots','training-progress');

% Training the network
[camo_clear_net2, info] = trainNetwork(CCtrainds,layers,trainOpts);

% Making predictions for Clear Images
ClearPreds = classify(camo_clear_net2,Cleartestds);
truetest = CleartestImgs.Labels;
nnz(ClearPreds == truetest)/numel(ClearPreds)
confusionchart(truetest,ClearPreds);
title('Clear Predictions')
% Making predictions for Camo Images
CamoPreds = classify(camo_clear_net2,Camotestds);
truetest = CamotestImgs.Labels;
nnz(CamoPreds == truetest)/numel(CamoPreds)
figure;
confusionchart(truetest,CamoPreds);
title('Camo Predictions')


