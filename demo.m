function [threeScores] = demo()
clear;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           %% Load the data.
%Please modify the dataset name that you want to test.
load(['WebKB_Texas.mat'],'fea','gt'); 
c = numel(unique(gt)); % The number of clusters
m = 5;%The number of anchors
opts.Distance = 'sqEuclidean';
k=5;
alpha = 0.1;

%%
tic;
Label = SONIC_fast(fea,c,k,m,alpha,opts);
toc;
tempScores = computeFourClusteringMetrics(Label,gt);
threeScores = [tempScores(1), tempScores(3), tempScores(4)];% % Save the scores of NMI, ACC, and PUR.
end