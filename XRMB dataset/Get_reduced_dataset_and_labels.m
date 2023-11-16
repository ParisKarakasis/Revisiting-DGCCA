clc; clear all; close all;

load('XRMBf2KALDI_window7_single1.mat')
load('XRMBf2KALDI_window7_single2.mat')

%% Center all samples per speaker
un_trainID = unique(trainID);
for i = 1:length(un_trainID)
    ind = (trainID==un_trainID(i)).*(1:length(trainID))';
    ind = ind(ind>0);
    Y1 = X1(ind,:);
    Y1 = Y1 - mean(Y1);
    X1(ind,:) = Y1;
    Y2 = X2(ind,:);
    Y2 = Y2 - mean(Y2);
    X2(ind,:) = Y2;
end

un_tuneID = unique(tuneID);
for i = 1:length(un_tuneID)
    ind = (tuneID==un_tuneID(i)).*(1:length(tuneID))';
    ind = ind(ind>0);
    Y1 = XV1(ind,:);
    Y1 = Y1 - mean(Y1);
    XV1(ind,:) = Y1;
    Y2 = XV2(ind,:);
    Y2 = Y2 - mean(Y2);
    XV2(ind,:) = Y2;
end

un_testID = unique(testID);
for i = 1:length(un_testID)
    ind = (testID==un_testID(i)).*(1:length(testID))';
    ind = ind(ind>0);
    Y1 = XTe1(ind,:);
    Y1 = Y1 - mean(Y1);
    XTe1(ind,:) = Y1;
    Y2 = XTe2(ind,:);
    Y2 = Y2 - mean(Y2);
    XTe2(ind,:) = Y2;
end
    
%% Restrict the dataset to the first 5 speakers
ind = (trainID<6).*(1:length(trainID))';
ind = ind(ind>0);

X1 = X1(ind,:);
X2 = X2(ind,:);
trainLabel = trainLabel(ind,:);

%% Based on the labels, create the third views with one-hot encoding
X3 = zeros(length(ind), 40);
for i = 1:length(ind)
    X3(i,trainLabel(i)+1) = 1;
end

XV3 = zeros(size(XV1,1), 40);
for i = 1:size(XV1,1)
    XV3(i,tuneLabel(i)+1) = 1;
end

XTe3 = zeros(size(XTe1,1), 40);
for i = 1:size(XTe1,1)
    XTe3(i,testLabel(i)+1) = 1;
end

save('XRMB_reduced','X1','X2','X3','XV1','XV2','XV3', 'XTe1', 'XTe2', 'XTe3', 'trainLabel', 'tuneLabel', 'testLabel')

disp("done")