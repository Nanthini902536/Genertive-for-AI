%% Camo FS Region Transformations (SoftMax)
% Passing clear and camo images through ClearNet to visualize the geometric
% transformations camouflage causes to animal regions in the feature space.
% Activations are extracted from the SoftMax layer in this case

%% MDS

% Loading previously trained networks and datasets
load('clear_net.mat')
load('CamoTestds2.mat')
load('ClearTestds2.mat')

% Extracting activations from FC (23rd) Layer
layer = 'prob';
Clear_netClearfeaturesTest = activations(clear_net,ClearTestds2,layer,'OutputAs','rows');
Clear_netCamofeaturesTest = activations(clear_net,CamoTestds2,layer,'OutputAs','rows');

% Extracting and Combining Bear Matrix Values
ClearBear = Clear_netClearfeaturesTest(1:37,:);
CamoBear = Clear_netCamofeaturesTest(1:30,:);
CombBears = [ClearBear;CamoBear];

% Extracting and Combining Canine Matrix Values
ClearCanine = Clear_netClearfeaturesTest(87:130,:);
CamoCanine = Clear_netCamofeaturesTest(87:130,:);
CombCanines = [ClearCanine;CamoCanine];

% Extracting and Combining Frog Matrix Values
ClearFrog = Clear_netClearfeaturesTest(205:244,:);
CamoFrog = Clear_netCamofeaturesTest(352:409,:);
CombFrogs = [ClearFrog;CamoFrog];

% Euclidean Distance Calculation for Bear MDS
Dist1 = NaN(67,67);
for i = 1:67
    for j = 1:67
        Dist1(i,j) = sqrt(sum((CombBears(i,:)-CombBears(j,:)).^2,2));
    end
end

% Euclidean Distance Calculation for Bear MDS
Dist2 = NaN(88,88);
for i = 1:88
    for j = 1:88
        Dist2(i,j) = sqrt(sum((CombCanines(i,:)-CombCanines(j,:)).^2,2));
    end
end

% Euclidean Disance Calculation for Frog MDS
Dist3 = NaN(98,98);
for i = 1:98
    for j = 1:98
        Dist3(i,j) = sqrt(sum((CombFrogs(i,:)-CombFrogs(j,:)).^2,2));
    end
end

% Bear MDS
BearMDSActs = mdscale(Dist1,5);

% Canine MDS 
CanineMDSActs = mdscale(Dist2,5);

% Frog MDS
FrogMDSActs = mdscale(Dist3,3);

% Plotting Clear and Camo Bear Clusters
figure;
hold on
plot3(BearMDSActs(1:37,1),BearMDSActs(1:37,2),BearMDSActs(1:37,3),'b*'); % Clear Bear
plot3(BearMDSActs(38:67,1),BearMDSActs(38:67,2),BearMDSActs(38:67,3),'bd'); % Camo Bear
title('ClearNet Bear Test SoftMax Activations MDS')
legend('Clear','Camo')

% Plotting Clear and Camo Canine Clusters
figure;
hold on
plot3(CanineMDSActs(1:44,1),CanineMDSActs(1:44,2),CanineMDSActs(1:44,3),'r*'); % Clear Canine
plot3(CanineMDSActs(45:88,1),CanineMDSActs(45:88,2),CanineMDSActs(45:88,3),'rd'); % Camo Canine
title('ClearNet Canine Test SoftMax Activations MDS')
legend('Clear','Camo')

% Plotting Clear and Camo Frog Clusters
figure;
hold on
plot3(FrogMDSActs(1:40,1),FrogMDSActs(1:40,2),FrogMDSActs(1:40,3),'g*'); % Clear Frog
plot3(FrogMDSActs(41:98,1),FrogMDSActs(41:98,2),FrogMDSActs(41:98,3),'gd'); % Camo Frog
title('ClearNet Frog Test SoftMax Activations MDS')
legend('Clear','Camo')

% Establishing Bear Clusters (Post MDS)
ClearBearActs = BearMDSActs(1:37,1:2);
CamoBearActs = BearMDSActs(38:67,1:2);

% Establishing Canine Clusters (Post MDS)
ClearCanineActs = CanineMDSActs(1:44,1:2);
CamoCanineActs = CanineMDSActs(45:88,1:2);

% Establishing Frog Clusters (Post MDS)
ClearFrogActs = FrogMDSActs(1:40,1:2);
CamoFrogActs = FrogMDSActs(41:98,1:2);

% Clear Bear True Center
for x = -0.2:0.1:0 % cluster borders on x-axis
    for y = -0.1:0.05:0.1 % cluster borders on y-axis
        SubtBear = (Clear_netClearfeaturesTest(1:37,1)) - [1.7 0.1];
        SqrBear = SubtBear .^ 2;
        SumBear = sum(SqrBear,2);
        DistBear = sqrt(SumBear);
        %figure;
        plot(DistBear,Clear_netClearfeaturesTest(1:37,1),'r*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (0,0) Appears to be the true center
% Camo Bear True Center
for x = 1.3:0.1:1.9 % cluster borders on x-axis
    for y = -0.2:0.1:0.4 % cluster borders on y-axis
        SubtBear = (Clear_netCamofeaturesTest(1:30,1)) - [1.7 0.1];
        SqrBear = SubtBear .^ 2;
        SumBear = sum(SqrBear,2);
        DistBear = sqrt(SumBear);
        figure;
        plot(DistBear,Clear_netCamofeaturesTest(1:30,1),'b*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
%(1.7,0.1)

% Calculating Distance Between Camo Bear Region and Clear Bear True Center
SubtBear2 = (CamoBearActs(:,:)) - [0 0];
SqrBear2 = SubtBear2 .^ 2;
SumBear2 = sum(SqrBear2,2);
DistBear2 = sqrt(SumBear2);
figure;
plot(DistBear2,CamoBearActs(:,1),'bd')
xlabel('Distance')
ylabel('Activations')
title('Bear Region Distance from Clear Bear Region True Center')
legend('Camo','Clear')

% Clear bear distance from camo bear true center
SubtBear = (ClearBearActs(:,:)) - [1.7 0.1];
SqrBear = SubtBear .^ 2;
SumBear = sum(SqrBear,2);
DistBear = sqrt(SumBear);
%figure;
plot(DistBear,ClearBearActs(:,1),'r*')
% Clear Canine True Center
for x = -0.6:0.1:0.1 % cluster borders on x-axis
    for y = -0.1:0.05:0.1 % cluster borders on y-axis
        SubtCanine = (ClearCanineActs(:,:)) - [x y];
        SqrCanine = SubtCanine .^ 2;
        SumCanine = sum(SqrCanine,2);
        DistCanine = sqrt(SumCanine);
        figure;
        plot(DistCanine,ClearCanineActs(:,1),'r*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (0.1,-0.1) Appears to be the true center

% Calculating Distance Between Camo Canine Region and Clear Canine True Center
SubtCanine2 = (CamoCanineActs(:,:)) - [0.1 -0.1];
SqrCanine2 = SubtCanine2 .^ 2;
SumCanine2 = sum(SqrCanine2,2);
DistCanine2 = sqrt(SumCanine2);
figure;
plot(DistCanine2,CamoCanineActs(:,1),'rd')
xlabel('Distance')
ylabel('Activations')
title('Canine Region Distance from Clear Canine Region True Center')
legend('Camo','Clear')

% Clear Frog True Center
for x = -0.4:0.1:0.2 % cluster borders on x-axis
    for y = -0.1:0.05:0.1 % cluster borders on y-axis
        SubtFrog = (ClearFrogActs(:,:)) - [x y];
        SqrFrog = SubtFrog .^ 2;
        SumFrog = sum(SqrFrog,2);
        DistFrog = sqrt(SumFrog);
        figure;
        plot(DistFrog,ClearFrogActs(:,1),'g*')
        title(['x= ' num2str(x) 'y= ' num2str(y)])
    end
end
% (0.2,0) appears to be the true center

% Calculating Distance Between Camo Frog Region and Clear Frog True Center
SubtFrog2 = (CamoFrogActs(:,:)) - [0.2 0];
SqrFrog2 = SubtFrog2 .^ 2;
SumFrog2 = sum(SqrFrog2,2);
DistFrog2 = sqrt(SumFrog2);
figure;
plot(DistFrog2,CamoFrogActs(:,1),'gd')
xlabel('Distance')
ylabel('Activations')
title('Frog Region Distance from Clear Frog Region True Center')
legend('Camo','Clear')


