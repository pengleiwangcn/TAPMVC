clc
clear
close all
addpath('funs');

load(fullfile('data', 'MSRCv1.mat'));
trueLabels = double(Y(:));
n = numel(trueLabels);
K = numel(unique(trueLabels));
numAnchors = min(500, max(50, 5 * K));
options.cosine = true;


%% The best config
rho = 0.15;
% 3sources:  rho = 0.40
% MSRCv1: rho = 0.15
% WikipediaArticles_total: rho = 0.9
% YouTubeFace10_4Views: rho = 0.25
% YouTubeFace20_4Views: rho = 0.3


%% preprocessing
for v = 1:numel(X)
    X{v} = full(double(X{v}));
    X{v} = X{v} - mean(X{v}, 1);
    sigma = std(X{v}, 1, 1);
    sigma(sigma == 0) = 1;
    X{v} = X{v} ./ sigma;
end


%% optimization
[W, anchorIndices] = generate_shared_anchor_graph(X, numAnchors, 5);
Views = cellfun(@(x) x / sqrt(size(x, 2)), X, 'UniformOutput', false);
jointFeatures = cat(2, Views{:});
F0 = full(sparse((1:n)', Init_partition(jointFeatures, K, options.cosine), 1, n, K));
lambdaF0 = full(sparse((1:n)', rgs_kmeans_partition(jointFeatures, anchorIndices, K, options.cosine), 1, n, K));
lambdaSmooth = compute_lambda_smooth(X, W, lambdaF0, options.cosine);
options.lambda = rho * lambdaSmooth;
F = main(X, W, F0, options);
[~, labels] = max(F, [], 2);


%% evaluation
metrics = ClusteringEvaluation7(trueLabels, labels);
fprintf('samples=%d, views=%d, clusters=%d, anchors=%d\n', n, numel(X), K, numAnchors);
fprintf('rho=%.3g, lambda=%.6g\n', rho, options.lambda);
fprintf('%8s %8s %8s %8s %8s %8s %8s\n', 'ACC', 'NMI', 'Purity', 'F1', 'Precision', 'Recall', 'ARI');
fprintf('%8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f\n', metrics);