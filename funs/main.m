function [F, AF, AG, C, obj, info] = main(X, W, F, options)
% J = sum(AF.*EF,'all') + lambda*sum(AG.*EG,'all').
if nargin < 4
    options = struct;
end
defaults = struct('max_iter', 200, 'epsilon', 1e-6, 'verbose', true, 'cosine', false);
fields = fieldnames(defaults);
for j = 1:numel(fields)
    if ~isfield(options, fields{j})
        options.(fields{j}) = defaults.(fields{j});
    end
end
assert(isfield(options, 'lambda') && isscalar(options.lambda) && isfinite(options.lambda) && options.lambda > 0, 'Set options.lambda in the calling script before optimization.');
cosine = logical(options.cosine);
distanceMetric = 'euclidean';
if cosine
    distanceMetric = 'cosine';
end
lambda = options.lambda;
info = struct('lambda', lambda, 'cosine', cosine, 'distanceMetric', distanceMetric);
if options.verbose
    fprintf('lambda=%.6g\n', lambda);
end

V = numel(X);
[n, K] = size(F);
if cosine
    for v = 1:V
        scale = sqrt(sum(X{v}.^2, 2));
        scale(scale == 0) = 1;
        X{v} = X{v} ./ scale;
    end
end
xnorm = cellfun(@(x) sum(x.^2, 2), X, 'UniformOutput', false);

AF = ones(V, K);
AG = AF;
C = cellfun(@(x) zeros(K, size(x, 2)), X, 'UniformOutput', false);
[C, delta] = centers(X, F, C, xnorm);

degree = zeros(n, V);
m = size(W{1}, 2);
anchorMass = zeros(m, V);
for v = 1:V
    assert(size(W{v}, 2) == m, 'All views must share the same anchor identities.');
    degree(:, v) = full(sum(W{v}, 2));
    anchorMass(:, v) = full(sum(W{v}, 1))';
end
[H, Q, hEnergy] = shared_membership(W, F, AG, anchorMass);
[EF, EG] = losses(F, delta, Q, degree, hEnergy);
obj = zeros(options.max_iter + 1, 1);
obj(1) = objective(AF, EF, AG, EG, lambda);

for iter = 1:options.max_iter
    [C, delta] = centers(X, F, C, xnorm);
    [EF, ~] = losses(F, delta, Q, degree, hEnergy);
    AF = preference(EF, options.epsilon);

    [H, Q, hEnergy] = shared_membership(W, F, AG, anchorMass);
    [~, EG] = losses(F, delta, Q, degree, hEnergy);
    AG = preference(EG, options.epsilon);

    fitCost = zeros(n, K);
    neighbor = zeros(n, K);
    for v = 1:V
        fitCost = fitCost + delta{v} .* AF(v, :);
        neighbor = neighbor + Q{v} .* AG(v, :);
    end
    F = solve_simplex_rows(lambda * (degree * AG), lambda * neighbor - fitCost / 2);

    [EF, EG] = losses(F, delta, Q, degree, hEnergy);
    obj(iter + 1) = objective(AF, EF, AG, EG, lambda);
    decrease = (obj(iter) - obj(iter + 1)) / obj(iter);
    if options.verbose
        fprintf('Iter %3d: objective = %.10g, relative change = %.3e\n', iter, obj(iter + 1), decrease);
    end
    if decrease < options.epsilon
        break
    end
end
obj = obj(1:iter + 1);
info.H = H;
end

function [H, Q, hEnergy] = shared_membership(W, F, AG, anchorMass)
V = numel(W);
K = size(F, 2);
score = zeros(size(anchorMass, 1), K);
for v = 1:V
    score = score + (W{v}' * F) .* AG(v, :);
end
% The common positive lambda cancels from the H subproblem.
H = solve_simplex_rows(anchorMass * AG, score);
H(all(anchorMass == 0, 2), :) = 1 / K;
Q = cell(1, V);
hEnergy = zeros(V, K);
for v = 1:V
    Q{v} = W{v} * H;
    hEnergy(v, :) = sum(anchorMass(:, v) .* H.^2, 1);
end
end

function [EF, EG] = losses(F, delta, Q, degree, hEnergy)
V = numel(delta);
EF = zeros(V, size(F, 2));
EG = EF;
for v = 1:V
    EF(v, :) = sum(F .* delta{v}, 1);
    EG(v, :) = max(sum(degree(:, v) .* F.^2, 1) - 2 * sum(F .* Q{v}, 1) + hEnergy(v, :), 0);
end
end

function A = preference(E, epsilon)
safe = E + epsilon;
geometricMean = exp(mean(log(safe), 1));
A = geometricMean ./ safe;
% Remove floating-point drift in the product-to-one constraint.
A = A ./ exp(mean(log(A), 1));
assert(all(isfinite(A), 'all') && all(A > 0, 'all'), 'Invalid product-normalized preferences.');
end

function value = objective(AF, EF, AG, EG, lambda)
value = sum(AF .* EF, 'all') + lambda * sum(AG .* EG, 'all');
end

function [C, delta] = centers(X, F, C, xnorm)
clusterMass = sum(F, 1)';
active = clusterMass > 0;
delta = cell(size(X));
for v = 1:numel(X)
    weightedSum = F' * X{v};
    C{v}(active, :) = weightedSum(active, :) ./ clusterMass(active);
    delta{v} = max(xnorm{v} + sum(C{v}.^2, 2)' - 2 * X{v} * C{v}', 0);
end
end

function F = solve_simplex_rows(a, score)
% Minimize sum(a.*F.^2 - 2*score.*F) subject to simplex rows.
[n, K] = size(score);
F = zeros(n, K);
linear = all(a == 0, 2);
rows = find(linear);
if ~isempty(rows)
    [~, best] = max(score(rows, :), [], 2);
    F(sub2ind([n, K], rows, best)) = 1;
end

rows = find(~linear);
if isempty(rows)
    return
end
a = a(rows, :);
score = score(rows, :);
score = score - max(score, [], 2);
[sortedScore, order] = sort(score, 2, 'descend');
nr = numel(rows);
sortedA = a((1:nr)' + (order - 1) * nr);
thresholds = (cumsum(sortedScore ./ sortedA, 2) - 1) ./ cumsum(1 ./ sortedA, 2);
lastActive = sum(sortedScore > thresholds, 2);
tau = thresholds(sub2ind([nr, K], (1:nr)', lastActive));
f = max(score - tau, 0) ./ a;
F(rows, :) = f ./ sum(f, 2);
end
