%% Camouflage PCA Cluster Measuring
% Plotting the centroid of each animal cluster, and calculating the
% distance of each point from that cluster.

%% Plotting Feature Space

% Loading previously trained networks
load('camo_net.mat')
load('clear_net.mat')

% Loading Augmented Image Datastores used for training networks
load('ClearTestds2.mat')
load('CamoTestds2.mat')

% Extracting activations from FC (23rd) Layer
layer = 'fc';
Camo_netCamofeaturesTest = activations(camo_net,CamoTestds2,layer,'OutputAs','rows');
Camo_netClearfeaturesTest = activations(camo_net,ClearTestds2,layer,'OutputAs','rows');
Clear_netClearfeaturesTest = activations(clear_net,ClearTestds2,layer,'OutputAs','rows');
Clear_netCamofeaturesTest = activations(clear_net,CamoTestds2,layer,'OutputAs','rows');

% Reducing Dimensionality
[coeff1,score1] = pca(Camo_netCamofeaturesTest);
[coeff2,score2] = pca(Camo_netClearfeaturesTest);
[coeff3,score3] = pca(Clear_netClearfeaturesTest);
[coeff4,score4] = pca(Clear_netCamofeaturesTest);
camo_net_Camotest_acts = score1(:,1:2);
camo_net_Cleartest_acts = score2(:,1:2);
clear_net_Cleartest_acts = score3(:,1:2);
clear_net_Camotest_acts = score4(:,1:2);

% 3D Activation Bar Graph
% camo_net_Camotest_acts = score1(:,1:3);
% camo_net_Cleartest_acts = score2(:,1:3);
% clear_net_Cleartest_acts = score3(:,1:3);
% clear_net_Camotest_acts = score4(:,1:3);
% figure;
% bar3(CamoBear(:,:))

% Assigning Camo Animal Clusters
CamoBear = camo_net_Camotest_acts(1:30,:);
CamoCan = camo_net_Camotest_acts(143:185,:);
CamoFrog = camo_net_Camotest_acts(352:409,:);
CamoRept = camo_net_Camotest_acts(546:598,:);

% Plotting Camo Animal Group Clusters
figure;
hold on
plot(CamoBear(:,1),CamoBear(:,2),'bd')
plot(CamoCan(:,1),CamoCan(:,2),'rd')
plot(CamoFrog(:,1),CamoFrog(:,2),'gd')
plot(CamoRept(:,1),CamoRept(:,2),'blackd')
title('Camo BCFR Test Activations (PCA)')
plot(6,0,'b*')
plot(5,0,'r*')
plot(-22,-4,'g*')
plot(-22,-3,'black*')
legend('Bear','Canine','Frog','Reptile','Bear Center','Canine Center','Frog Center','Reptile Center')

% Assigning Clear Animal Clustes
ClearBear = clear_net_Cleartest_acts(1:37,:);
ClearCan = clear_net_Cleartest_acts(87:130,:);
ClearFrog = clear_net_Cleartest_acts(205:244,:);
ClearRept = clear_net_Cleartest_acts(371:396,:);

% Plotting Clear Animal Group Clusters
figure;
hold on
plot(ClearBear(:,1),ClearBear(:,2),'b*')
plot(ClearCan(:,1),ClearCan(:,2),'r*')
plot(ClearFrog(:,1),ClearFrog(:,2),'g*')
plot(ClearRept(:,1),ClearRept(:,2),'black*')
title('Clear BCFR Test Activations')
plot(25,6,'bd')
plot(26,-1,'rd')
plot(-3,4,'gd')
plot(1,0,'blackd')
legend('Bear','Canine','Frog','Reptile','Bear Center','Canine Center','Frog Center','Reptile Center')

% Calculating Camo Cluster Centroids
BearMean = [mean(CamoBear(:,1)),mean(CamoBear(:,2))];
CanMean = [mean(CamoCan(:,1)),mean(CamoCan(:,2))];
FrogMean = [mean(CamoFrog(:,1)),mean(CamoFrog(:,2))];
ReptMean = [mean(CamoRept(:,1)),mean(CamoRept(:,2))];

% Calculating Clear Cluster Centroids
BearMean2 = [mean(ClearBear(:,1)),mean(ClearBear(:,2))];
CanMean2 = [mean(ClearCan(:,1)),mean(ClearCan(:,2))];
FrogMean2 = [mean(ClearFrog(:,1)),mean(ClearFrog(:,2))];
ReptMean2 = [mean(ClearRept(:,1)),mean(ClearRept(:,2))];

% Calculating Centroid Distance (Bear, Camo Testing)
SubtBear = CamoBear - BearMean;
SqrBear = SubtBear .^ 2;
SumBear = sum(SqrBear,2);
DistBear = sqrt(SumBear);

% Calculating Centroid Distance (Canine, Camo Testing)
SubtCan = CamoCan - CanMean;
SqrCan = SubtCan .^ 2;
SumCan = sum(SqrCan,2);
DistCan = sqrt(SumCan);

% Calculating Centroid Distance (Frog, Camo Testing)
SubtFrog = CamoFrog - FrogMean;
SqrFrog = SubtFrog .^ 2;
SumFrog = sum(SqrFrog,2);
DistFrog = sqrt(SumFrog);

% Calculating Centroid Distance (Reptile, Camo Testing)
SubtRept = CamoRept - ReptMean;
SqrRept = SubtRept .^ 2;
SumRept = sum(SqrRept,2);
DistRept = sqrt(SumRept);

% Calculating Centroid Distance (Bear, Clear Testing)
SubtBear2 = ClearBear - BearMean2;
SqrBear2 = SubtBear2 .^ 2;
SumBear2 = sum(SqrBear2,2);
DistBear2 = sqrt(SumBear2);

% Calculating Centroid Distance (Canine, Clear Testing)
SubtCan2 = ClearCan - CanMean2;
SqrCan2 = SubtCan2 .^ 2;
SumCan2 = sum(SqrCan2,2);
DistCan2 = sqrt(SumCan2);

% Calculating Centroid Distance (Frog, Clear Testing)
SubtFrog2 = ClearFrog - FrogMean2;
SqrFrog2 = SubtFrog2 .^ 2;
SumFrog2 = sum(SqrFrog2,2);
DistFrog2 = sqrt(SumFrog2);

% Calculating Centroid Distance (Reptile, Clear Testing)
SubtRept2 = ClearRept - ReptMean2;
SqrRept2 = SubtRept2 .^ 2;
SumRept2 = sum(SqrRept2,2);
DistRept2 = sqrt(SumRept2);

% Plotting Against Activations (Camo Testing)
figure;
hold on
plot(DistBear,CamoBear(:,1),'bd');
plot(DistCan,CamoCan(:,1),'rd');
plot(DistFrog,CamoFrog(:,1),'gd');
plot(DistRept,CamoRept(:,1),'blackd');
legend('Bear','Canine','Frog','Reptile')

% Plotting Against Activations Before PCA (Camo Testing)
figure;
hold on
plot(DistBear,Camo_netCamofeaturesTest(1:30,1),'bd');
plot(DistCan,CamoCan(:,1),'rd');
plot(DistFrog,CamoFrog(:,1),'gd');
plot(DistRept,CamoRept(:,1),'blackd');
legend('Bear','Canine','Frog','Reptile')
% Adjusted for pre-PCA values

% Plotting Against Activations (Clear Testing)
figure;
hold on
plot(DistBear2,ClearBear(:,1),'b*');
plot(DistCan2,ClearCan(:,1),'r*');
plot(DistFrog2,ClearFrog(:,1),'g*');
plot(DistRept2,ClearRept(:,1),'black*');
legend('Bear','Canine','Frog','Reptile')


% for x = 23:1:27 %borders of cluster on x-axis
%     for y = 7:1:11 %borders of cluster on y-axis
%         SubtC_Bear = (red_clear_acts(1:93,:)) - [x y];
%         SqrC_Bear = SubtC_Bear .^ 2;
%         SumC_Bear = sum(SqrC_Bear,2);
%         DistC_Bear = sqrt(SumC_Bear);
%         figure;
%         plot(DistC_Bear,red_clear_acts(1:93,1),'r*')
%         title(['x= ' num2str(x) 'y= ' num2str(y)])
%     end
% end

% Camo Bear True Center  
for x = 3:1:6 %borders of cluster on x-axis 
    for y = -2:1:0 %borders of cluster on y-axis
        SubtBear = (CamoBear(:,:)) - [x y];
        SqrBear = SubtBear .^ 2;
        SumBear = sum(SqrBear,2);
        DistBear = sqrt(SumBear);
        figure;
        plot(DistBear,CamoBear(:,1),'b*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (6,0) appears to be the true center

% Clear Bear True Center
for x = 23:1:25 %borders of cluster on x-axis
    for y = 5:1:7 %borders of cluster on y-axis
        SubtBear2 = (ClearBear(:,:)) - [x y];
        SqrBear2 = SubtBear2 .^ 2;
        SumBear2 = sum(SqrBear2,2);
        DistBear2 = sqrt(SumBear2);
        figure;
        plot(DistBear2,ClearBear(:,1),'bd')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (25,6) appears to be true center

% Camo Canine True Center
for x = 3:1:6 %borders of cluster on x-axis
    for y = -2:1:2 %borders of cluster on y-axis
        SubtCan = (CamoCan(:,:)) - [x y];
        SqrCan = SubtCan .^ 2;
        SumCan = sum(SqrCan,2);
        DistCan = sqrt(SumCan);
        figure;
        plot(DistCan,CamoCan(:,1),'r*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (5,0) appears to be the true center

% Clear Canine True Center
for x = 24:0.5:26 %borders of cluster on x-axis
    for y = -1:0.5:1 %borders of cluster on y-axis
        SubtCan2 = (ClearCan(:,:)) - [x y];
        SqrCan2 = SubtCan2 .^ 2;
        SumCan2 = sum(SqrCan2,2);
        DistCan2 = sqrt(SumCan2);
        figure;
        plot(DistCan2,ClearCan(:,1),'r*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (26,-1) appears to be true center

% Camo Frog True Center
for x = 21:0.5:22 %borders of cluster on x-axis
    for y = -5:0.5:-4 %borders of cluster on y-axis
        SubtFrog = (CamoFrog(:,:)) - [x y];
        SqrFrog = SubtFrog .^ 2;
        SumFrog = sum(SqrFrog,2);
        DistFrog = sqrt(SumFrog);
        figure;
        plot(DistFrog,CamoFrog(:,1),'g*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (22,-4) appears to be the true center

% Clear Frog True Center
for x = -6:1:-3 %borders of cluster on x-axis
    for y = 3:1:7 %borders of cluster on y-axis
        SubtFrog2 = (ClearFrog(:,:)) - [x y];
        SqrFrog2 = SubtFrog2 .^ 2;
        SumFrog2 = sum(SqrFrog2,2);
        DistFrog2 = sqrt(SumFrog2);
        figure;
        plot(DistFrog2,ClearFrog(:,1),'g*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (-3,4) appears to be true center

% Camo Reptile True Center
for x = 20:0.5:22
    for y = -5:0.5:-3
        SubtRept = (CamoRept(:,:)) - [x y];
        SqrRept = SubtRept .^ 2;
        SumRept = sum(SqrRept,2);
        DistRept = sqrt(SumRept);
        figure;
        plot(DistRept,CamoRept(:,1),'black*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (22, -3) appears to be the true center

% Clear Reptile True Center
for x = -1:0.5:1
    for y = 0:1:3
        SubtRept2 = (ClearRept(:,:)) - [x y];
        SqrRept2 = SubtRept2 .^ 2;
        SumRept2 = sum(SqrRept2,2);
        DistRept2 = sqrt(SumRept2);
        figure;
        plot(DistRept2,ClearRept(:,1),'black*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
%(1,0) appears to be true center

% Combining CamoNet Camo BCFR Distance v Activations plots (PCA)
figure;
hold on
SubtBear = (CamoBear(:,:)) - [6 0];
SqrBear = SubtBear .^ 2;
SumBear = sum(SqrBear,2);
DistBear = sqrt(SumBear);
plot(DistBear,CamoBear(:,1),'b*')
SubtCan = (CamoCan(:,:)) - [5 0];
SqrCan = SubtCan .^ 2;
SumCan = sum(SqrCan,2);
DistCan = sqrt(SumCan);
plot(DistCan,CamoCan(:,1),'r*')
SubtFrog = (CamoFrog(:,:)) - [22 -4];
SqrFrog = SubtFrog .^ 2;
SumFrog = sum(SqrFrog,2);
DistFrog = sqrt(SumFrog);
plot(DistFrog,CamoFrog(:,1),'g*')
SubtRept = (CamoRept(:,:)) - [22 -3];
SqrRept = SubtRept .^ 2;
SumRept = sum(SqrRept,2);
DistRept = sqrt(SumRept);
plot(DistRept,CamoRept(:,1),'black*')
xlabel('Distance')
ylabel('Activations')
title('CamoNet Camo BCFR: Distance vs Activations (PCA)')
legend('Bear','Canine','Frog','Reptile')











