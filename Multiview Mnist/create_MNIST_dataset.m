clc; clear all; close all;

load('mnist_all.mat')

A = [];
labels = [];
A = [A; train0];
labels = [labels; 0*ones(size(train0,1),1)];
A = [A; train1];
labels = [labels; 1*ones(size(train1,1),1)];
A = [A; train2];
labels = [labels; 2*ones(size(train2,1),1)];
A = [A; train3];
labels = [labels; 3*ones(size(train3,1),1)];
A = [A; train4];
labels = [labels; 4*ones(size(train4,1),1)];
A = [A; train5];
labels = [labels; 5*ones(size(train5,1),1)];
A = [A; train6];
labels = [labels; 6*ones(size(train6,1),1)];
A = [A; train7];
labels = [labels; 7*ones(size(train7,1),1)];
A = [A; train8];
labels = [labels; 8*ones(size(train8,1),1)];
A = [A; train9];
labels = [labels; 9*ones(size(train9,1),1)];
X = A;
A = [];
labels_Te = [];
A = [A; test0];
labels_Te = [labels_Te; 0*ones(size(test0,1),1)];
A = [A; test1];
labels_Te = [labels_Te; 1*ones(size(test1,1),1)];
A = [A; test2];
labels_Te = [labels_Te; 2*ones(size(test2,1),1)];
A = [A; test3];
labels_Te = [labels_Te; 3*ones(size(test3,1),1)];
A = [A; test4];
labels_Te = [labels_Te; 4*ones(size(test4,1),1)];
A = [A; test5];
labels_Te = [labels_Te; 5*ones(size(test5,1),1)];
A = [A; test6];
labels_Te = [labels_Te; 6*ones(size(test6,1),1)];
A = [A; test7];
labels_Te = [labels_Te; 7*ones(size(test7,1),1)];
A = [A; test8];
labels_Te = [labels_Te; 8*ones(size(test8,1),1)];
A = [A; test9];
labels_Te = [labels_Te; 9*ones(size(test9,1),1)];
X_Te = A;

clear A;
X = double(X);
for i = 1:size(X,1)
    tmp = X(i,:);
    tmp = reshape(tmp,28,28);
    X(i,:) = tmp(:);
end

X_Te = double(X_Te);
for i = 1:size(X_Te,1)
    tmp = X_Te(i,:);
    tmp = reshape(tmp,28,28);
    X_Te(i,:) = tmp(:);
end

save('MNIST_dataset','X','X_Te','labels','labels_Te')
