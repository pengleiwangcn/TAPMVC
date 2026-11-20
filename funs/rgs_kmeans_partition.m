function labels = rgs_kmeans_partition(features, anchorIndices, K, cosine)
distanceMetric = 'sqeuclidean';
if cosine
    distanceMetric = 'cosine';
end
centers = features(anchorIndices(1:K), :);
labels = kmeans(features, K, 'Start', centers, 'Distance', distanceMetric, 'Replicates', 1, 'EmptyAction', 'singleton');
[~, ~, labels] = unique(labels(:));
assert(max(labels) == K, 'RGS-initialized k-means did not return K clusters.');
end
