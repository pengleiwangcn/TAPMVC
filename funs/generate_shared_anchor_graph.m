function [W, indices] = generate_shared_anchor_graph(X, m, k)
indices = rgs_indices(X, m);
W = cell(size(X));
for v = 1:numel(X)
    W{v} = anchor_graph(X{v}, X{v}(indices, :), k);
end
end

function Z = anchor_graph(X, anchors, k)
n = size(X, 1);
m = size(anchors, 1);
distances = pdist2(X, anchors, 'squaredeuclidean');
[nearest, order] = mink(distances, k + 1, 2);
gap = nearest(:, k + 1) - nearest(:, 1:k);
denominator = sum(gap, 2);
tied = denominator == 0;
denominator(tied) = 1;
weights = gap ./ denominator;
weights(tied, :) = 1 / k;
neighborIndex = order(:, 1:k);
rows = repmat((1:n)', 1, k);
Z = sparse(rows(:), neighborIndex(:), weights(:), n, m);
end

function indices = rgs_indices(X, m)
% Each view proposes {probe, representative} from its own residual regions.
% Shared acceptance uses external relative gain; cached distances update coverage.
n = size(X{1}, 1);
V = numel(X);
dimensions = cellfun(@(x) size(x, 2), X);
squaredNorm = cellfun(@(x) sum(x.^2, 2), X, 'UniformOutput', false);
initialScore = zeros(n, 1);
for v = 1:V
    initialScore = initialScore + sum((X{v} - mean(X{v}, 1)).^2, 2) / dimensions(v);
end
indices = zeros(m, 1);
chosen = false(n, 1);
[~, indices(1)] = min(initialScore);
chosen(indices(1)) = true;
residual = zeros(n, V);
owner = ones(n, V);
for v = 1:V
    residual(:, v) = sum((X{v} - X{v}(indices(1), :)).^2, 2) / dimensions(v);
end

for t = 1:m-1
    candidates = zeros(2 * V, 1);
    for v = 1:V
        if sum(residual(:, v)) == 0
            continue
        end
        energy = accumarray(owner(:, v), residual(:, v), [t, 1]);
        [~, region] = max(energy);
        members = find(owner(:, v) == region);
        [~, position] = max(residual(members, v));
        probe = members(position);

        distance = sum((X{v}(members, :) - X{v}(probe, :)).^2, 2) / dimensions(v);
        active = residual(members, v) > 0;
        weights = zeros(numel(members), 1);
        weights(active) = max(1 - distance(active) ./ residual(members(active), v), 0);
        support = members(weights > 0);
        weights = weights(weights > 0);
        center = (weights' * X{v}(support, :)) / sum(weights);
        distance = sum((X{v}(support, :) - center).^2, 2);
        if numel(support) > 1
            distance(support == probe) = inf;
        end
        [~, position] = min(distance);
        representative = support(position);
        candidates(2 * v - 1:2 * v) = [probe; representative];
    end

    candidates = unique(candidates(candidates > 0));
    candidates = candidates(~chosen(candidates));
    if isempty(candidates)
        available = find(~chosen);
        indices(t + 1:m) = available(1:m-t);
        break
    end

    q = numel(candidates);
    externalGain = zeros(q, 1);
    distanceCache = cell(V, 1);
    for v = 1:V
        distance = squaredNorm{v} + squaredNorm{v}(candidates)' - 2 * (X{v} * X{v}(candidates, :)');
        distance = max(distance, 0) / dimensions(v);
        distance(sub2ind([n, q], candidates, (1:q)')) = 0;
        distanceCache{v} = distance;
        improvement = max(residual(:, v) - distance, 0);
        active = residual(:, v) > 0;
        gain = sum(improvement(active, :) ./ residual(active, v), 1)';
        externalGain = externalGain + gain - double(residual(candidates, v) > 0);
    end

    [~, best] = max(externalGain);
    selected = candidates(best);
    indices(t + 1) = selected;
    chosen(selected) = true;
    for v = 1:V
        distance = distanceCache{v}(:, best);
        better = distance < residual(:, v);
        owner(better, v) = t + 1;
        residual(:, v) = min(residual(:, v), distance);
    end
end
end
