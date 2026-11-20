function lambdaSmooth = compute_lambda_smooth(X, W, F0, cosine)
if nargin < 4
    cosine = false;
end
V = numel(X);
[n, K] = size(F0);
if cosine
    for v = 1:V
        scale = sqrt(sum(X{v}.^2, 2));
        scale(scale == 0) = 1;
        X{v} = X{v} ./ scale;
    end
end
clusterMass = sum(F0, 1)';
active = clusterMass > 0;
pi = clusterMass' / n;
U = zeros(V, K);
G = U;
B = zeros(V, K);
T = zeros(V, 1);

m = size(W{1}, 2);
anchorMass = zeros(m, V);
score = zeros(m, K);
for v = 1:V
    anchorMass(:, v) = full(sum(W{v}, 1))';
    score = score + W{v}' * F0;
end
% With uniform initial preferences, weighted means already lie on the simplex.
totalMass = sum(anchorMass, 2);
unused = totalMass == 0;
totalMass(unused) = 1;
H0 = score ./ totalMass;
H0(unused, :) = 1 / K;

for v = 1:V
    mu = mean(X{v}, 1);
    centers = repmat(mu, K, 1);
    centers(active, :) = (F0(:, active)' * X{v}) ./ clusterMass(active);
    U(v, :) = sum(F0 .* pdist2(X{v}, centers, 'squaredeuclidean'), 1);
    T(v) = sum((X{v} - mu).^2, 'all');
    B(v, :) = clusterMass' .* sum((centers - mu).^2, 2)';
    degree = full(sum(W{v}, 2));
    Q = W{v} * H0;
    G(v, :) = sum(degree .* F0.^2, 1) - 2 * sum(F0 .* Q, 1) + sum(anchorMass(:, v) .* H0.^2, 1);
end
G = max(G, 0);
phiF0 = aggregate(U);
gamma = aggregate(G);
lambdaSmooth = inf;
if gamma == 0
    return
end

for t = (1:100) / 100
    Ut = max((1 - t) * U + t * (T * pi) + t * (1 - t) * B, 0);
    ratio = (aggregate(Ut) - phiF0) / (t * (2 - t) * gamma);
    lambdaSmooth = min(lambdaSmooth, ratio);
end
end

function value = aggregate(E)
% V times the sum of geometric means; a zero loss gives a zero product.
value = size(E, 1) * sum(exp(mean(log(E), 1)));
end
