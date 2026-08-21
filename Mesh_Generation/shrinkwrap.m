function [V, F] = shrinkwrap(P, opts)
% SHRINKWRAP  Shrink-wrap a convex hull onto a point cloud.
%
%   [V, F] = shrinkwrap(P)
%   [V, F] = shrinkwrap(P, opts)
%
%   Starts from the convex hull of point cloud P and iteratively
%   tightens the mesh toward the data, producing a watertight surface.
%
%   Inputs:
%       P       - Nx3 matrix of point cloud coordinates
%       opts    - (optional) struct with fields:
%           .nIter        - number of shrink iterations      (default: 80)
%           .alpha        - attraction strength toward data   (default: 0.6)
%           .beta         - Laplacian smoothing strength      (default: 0.3)
%           .nSubdiv      - subdivision passes before shrink  (default: 2)
%           .doPlot       - plot every plotEvery iters        (default: true)
%           .plotEvery    - plotting interval                 (default: 10)
%
%   Outputs:
%       V - Mx3 mesh vertices after shrink-wrapping
%       F - Kx3 mesh faces (triangles, 1-indexed)

%% --- Default options ---
if nargin < 2, opts = struct(); end
nIter     = getopt(opts, 'nIter',     80);
alpha     = getopt(opts, 'alpha',     0.6);
beta      = getopt(opts, 'beta',      0.3);
nSubdiv   = getopt(opts, 'nSubdiv',   2);
doPlot    = getopt(opts, 'doPlot',    true);
plotEvery = getopt(opts, 'plotEvery',  10);

%% --- Step 1: Convex hull ---
fprintf('Computing convex hull...\n');
F = convhull(P(:,1), P(:,2), P(:,3));
V = P(unique(F(:)), :);

% Reindex faces to new vertex list
[~, fmap] = ismember(F, unique(F(:)));
F = fmap;

%% --- Step 2: Subdivide for resolution ---
for s = 1:nSubdiv
    fprintf('Subdivision pass %d/%d  (%d verts, %d faces)\n', ...
            s, nSubdiv, size(V,1), size(F,1));
    [V, F] = subdivide_mesh(V, F);
end
fprintf('Mesh after subdivision: %d verts, %d faces\n', size(V,1), size(F,1));

%% --- Step 3: Build KD-tree for nearest-neighbor queries ---
kdt = KDTreeSearcher(P);

%% --- Step 4: Iterative shrink-wrap ---
if doPlot
    fig = figure('Name','Shrink-Wrap Progress','Color','w');
end

for iter = 1:nIter
    % --- 4a. Attraction: move each vertex toward nearest point ---
    idxNN = knnsearch(kdt, V);
    targets = P(idxNN, :);
    displacement = targets - V;

    % --- 4b. Laplacian smoothing ---
    L = compute_laplacian(V, F);

    % --- 4c. Update vertices ---
    %  new_V = V + alpha * attraction + beta * laplacian_smoothing
    V = V + alpha * displacement + beta * L;

    % --- 4d. Re-project onto nearest points (optional clamp) ---
    %  Prevents overshooting: don't move past the data
    idxNN2 = knnsearch(kdt, V);
    targets2 = P(idxNN2, :);
    dist_after = vecnorm(V - targets2, 2, 2);
    dist_before = vecnorm(displacement, 2, 2);

    % If a vertex overshot (got farther), snap it to nearest point
    overshot = dist_after > dist_before;
    V(overshot, :) = targets2(overshot, :);

    % --- 4e. Plot ---
    if doPlot && (mod(iter, plotEvery) == 0 || iter == 1)
        clf(fig);
        plot_state(P, V, F, iter, nIter);
    end

    fprintf('Iter %3d/%d  mean dist to cloud: %.4f\n', ...
        iter, nIter, mean(dist_after));
end

fprintf('Done. Final mesh: %d vertices, %d faces.\n', size(V,1), size(F,1));

end

%% ===================================================================
%  Helper functions
%  ===================================================================

function val = getopt(s, field, default)
    if isfield(s, field), val = s.(field); else, val = default; end
end

function [Vout, Fout] = subdivide_mesh(V, F)
% Loop-style midpoint subdivision: split each triangle into 4.
    nV = size(V, 1);
    nF = size(F, 1);
    edges = sort([F(:,[1 2]); F(:,[2 3]); F(:,[3 1])], 2);
    [uEdges, ~, edgeIdx] = unique(edges, 'rows');
    nE = size(uEdges, 1);

    % Midpoints
    midPts = 0.5 * (V(uEdges(:,1), :) + V(uEdges(:,2), :));
    Vout = [V; midPts];

    % New vertex indices for edge midpoints
    midIdx = nV + edgeIdx;
    m1 = midIdx(1:nF);          % midpoint of edge (v1,v2)
    m2 = midIdx(nF+1:2*nF);    % midpoint of edge (v2,v3)
    m3 = midIdx(2*nF+1:3*nF);  % midpoint of edge (v3,v1)

    v1 = F(:,1); v2 = F(:,2); v3 = F(:,3);

    Fout = [v1 m1 m3;
            m1 v2 m2;
            m3 m2 v3;
            m1 m2 m3];
end

function L = compute_laplacian(V, F)
% Uniform Laplacian displacement: for each vertex, average of neighbors minus vertex.
    nV = size(V, 1);

    % Build adjacency
    edges = [F(:,[1 2]); F(:,[2 3]); F(:,[3 1])];
    A = sparse(edges(:,1), edges(:,2), 1, nV, nV);
    A = A + A';              % symmetrize
    A = double(A > 0);      % binary adjacency

    deg = sum(A, 2);
    deg(deg == 0) = 1;      % avoid division by zero

    % Laplacian displacement = average(neighbors) - vertex
    neighborSum = A * V;
    L = neighborSum ./ deg - V;
end

function plot_state(P, V, F, iter, nIter)
    clf
    hold on;
    %scatter3(P(:,1), P(:,2), P(:,3), 10, [0.5 0.5 0.5], 'filled');
    trisurf(F, V(:,1), V(:,2), V(:,3), ...
        'FaceColor', [0.2 0.6 0.9], 'FaceAlpha', 1, ...
        'EdgeColor', [0.1 0.3 0.5], 'EdgeAlpha', 1, 'LineWidth',2);
    axis equal; grid on;

    view(21, -10)
    axis off
    % title(sprintf('Shrink-Wrap  —  Iteration %d / %d', iter, nIter));
    % xlabel('X'); ylabel('Y'); zlabel('Z');
    view(3);
    %camlight headlight; lighting gouraud;

    drawnow;
    disp('caca')

    ptCloud = pointCloud(V);

    normals = pcnormals(ptCloud, 18);
    ptCloud = pointCloud(V,'Normal',normals);

    ptCloud = pcdownsample(ptCloud,"gridAverage",20);
    [ptCloud,indices] = removeInvalidPoints(ptCloud);


    [mesh, depth, perVertexDensity] = pc2surfacemesh(ptCloud, "poisson", 6);

    removeDefects(mesh,"duplicate-vertices")
    removeDefects(mesh,"duplicate-faces")
    removeDefects(mesh,"unreferenced-vertices")
    removeDefects(mesh,"degenerate-faces")
    removeDefects(mesh,"nonmanifold-edges")

    % mesh = smoothSurfaceMesh(mesh, 10, 'Method','Taubin', 'ScaleFactor', [0.6 -0.63]);
    surfaceMeshShow(mesh)

    surfaceMeshShow(mesh, BackgroundColor="white", ColorMap=[0.8, 0.1, 0.1], alpha=1)

end