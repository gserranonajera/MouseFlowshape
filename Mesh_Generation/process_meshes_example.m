addpath(genpath('X:\Exchange\Guillermo\software\keller-lab-block-filetype\matlabWrapper'))

imdir = 'X:\mcdole\KM_18-04-10\Mmu_E1_RFPLifeAct_20180410_142505.corrected\Results\TimeFused\';
extension = '_CM00_CM01_CHN00.fusedStack.shifted.klb';

outdir = 'X:\mcdole\KM_18-04-10\analysis\000_flowshape\pre_mesh';
mkdir(outdir)

first_timepoint = 278;
skip_timepoints = 1;
max_timepoint = 287;
imfiles = dir([imdir filesep '**\*' extension]);
nT = length(imfiles);
resize_factor = 2;
scale = [1./resize_factor, 1./resize_factor, 1];
pixel_size = [0.4, 0.4, 2];
voxel_size = pixel_size./scale;

skip = 201;
n_std_int = 1;
clean_points = false;
final_triangulation = true;

for t = first_timepoint:skip_timepoints:max_timepoint
    %%
    tic
    fname = fullfile(imfiles(t).folder, imfiles(t).name);
    data = readKLBstack(fname);          % (X, Y, Z)
    data = data(1:resize_factor:end, 1:resize_factor:end,:);   % simple stride
    data = single(data);
    
    % threshold the image and extract the boundaries
    mean_data = mean(data(:));
    std_data = std(data(:));
    
    BW = data>mean_data+n_std_int*std_data;
    BW = bwareaopen(BW, 1e5);
    BW = imclose(BW, strel('disk', 5));
    BW = imgaussfilt(single(BW), 2) > 0.5;    
    BW = bwperim(BW);
        
    % get coordinates of boundaries
    [x, y, z] = ind2sub(size(BW), find(BW));
    
    %% Generate a cloud of points without internal structurures
    % reduce point cloud density
    pts = [x(1:skip:end).*voxel_size(1),...
           y(1:skip:end).*voxel_size(2),...
           z(1:skip:end).*voxel_size(3)];
    
    %% clean some outliers with single class support vector machines
    if clean_points
        pts_t = pts;
        removed = 1;
        gamma = 1 / (3 * var(pts_t(:)));
        gamma = gamma * 20;
        outliers = 0.05;
        max_it = 50;

        n_it = 0;
        while removed > 0

            % OneClassSVM en MATLAB
            model = fitcsvm(pts_t, ones(size(pts_t,1),1), ...
                'KernelFunction', 'rbf', ...
                'KernelScale', 1/sqrt(gamma), ...
                'Nu', outliers, ...
                'OutlierFraction', outliers);

            [~, scores] = predict(model, pts_t);
            s = scores(:,1);

            % Built-in outlier detection
            outlierMask = isoutlier(s, "median");  % or try 'gesd'
            fprintf('Removed %d points\n', removed);

            % Apply
            pts_clean = pts_t(~outlierMask, :);

            pts_t = pts_clean;
            if n_it > max_it; break; end
            removed = sum(outlierMask);
        end

        % clf
        % hold on
        % scatter3(pts(:,1), pts(:,2), pts(:,3), 'filled');
        % scatter3(pts_clean(:,1), pts_clean(:,2), pts_clean(:,3), 'filled')
    else
        pts_clean = pts;
    end

    %% Closing the embryonic cup 
    
    step = 21;
    dist2surf = 21;  % desired distance from surface
    
    % 1. We create the alpha shape that more tightly follows the surface
    % than the convex hull
    [~, vol_est] = convhull(pts_clean);
    rad = 0.001*((vol_est*3/4)./pi)^1/3;
    shp = alphaShape(pts_clean(:,1), pts_clean(:,2), pts_clean(:,3), rad, 'HoleThreshold', vol_est);
    
    % 2. We generate a meshgrid with the desire resolution
    xRange = min(pts_clean(:,1)):step:max(pts_clean(:,1));
    yRange = min(pts_clean(:,2)):step:max(pts_clean(:,2));
    zRange = min(pts_clean(:,3)):step:max(pts_clean(:,3));
    [X, Y, Z] = meshgrid(xRange, yRange, zRange);
    query_pts = [X(:), Y(:), Z(:)];
    
    % 3. We selecect those points inside the alpha shape
    inside = inShape(shp, query_pts(:,1), query_pts(:,2), query_pts(:,3));
    inner_pts = query_pts(inside, :);
    
    % 4. We compute distance from each inner point to the surface (original cloud)
    [~, D] = knnsearch(pts_clean, inner_pts);
    
    % 5. We Keep only points within a shell at distance from surface
    inner_pts = inner_pts(D > dist2surf, :);

    % clf
    % hold on
    % % plot(shp, 'FaceAlpha', 0.1, 'EdgeAlpha', 0); hold on;
    % scatter3(pts_clean(:,1), pts_clean(:,2), pts_clean(:,3), 'filled')
    % scatter3(inner_pts(:,1), inner_pts(:,2), inner_pts(:,3), 'filled');
    % axis equal;
    % title(sprintf('Points within %g units from surface', dist));


    %% Run shrink-wrap
    % Starts from the convex hull of point cloud P and iteratively tightens
    % the mesh toward the data, producing a watertight surface.
    % 
    pts_closed = [pts_clean; inner_pts];

    opts.nIter     = 50;   % iterations
    opts.alpha     = 0.5;   % attraction to point cloud
    opts.beta      = 0.3;   % Laplacian smoothing
    opts.nSubdiv   = 4;     % subdivision passes on initial hull
    opts.doPlot    = false;
    opts.plotEvery = 10;
    
    [pts, tri] = shrinkwrap(pts_closed, opts);

    TR = triangulation(tri, pts);
    F = faceNormal(TR);

    % 1. Clean unreferenced vertices first (fixes the warning)
    [usedIdx, ~, newIdx] = unique(tri(:));
    pts = pts(usedIdx, :);
    tri = newIdx(reshape(1:numel(tri), size(tri)));


    e1 = pts(tri(:,2),:) - pts(tri(:,1),:);
    e2 = pts(tri(:,3),:) - pts(tri(:,1),:);
    areas = 0.5 * vecnorm(cross(e1, e2, 2), 2, 2);

    % Remove zero-area faces
    tri(areas < 1e-10, :) = [];
    
    % Remove unreferenced vertices
    [usedIdx, ~, newIdx] = unique(tri(:));
    pts = pts(usedIdx, :);
    tri = reshape(newIdx, size(tri));
    
    % Recompute normals
    TR = triangulation(double(tri), double(pts));
    VN = vertexNormal(TR);
    
    % Check for bad normals again
    exact_z = abs(VN(:,1)) < 1e-10 & abs(VN(:,2)) < 1e-10 & abs(VN(:,3) - 1) < 1e-10;
    fprintf('%d vertices still have [0 0 1] normals\n', sum(exact_z));

    % 1. Merge near-coincident vertices
    [pts, ~, remap] = uniquetol(pts, 1e-6, 'ByRows', true);
    tri = remap(tri);
    
    % Remove collapsed faces (where merge made 2+ indices the same)
    collapsed = tri(:,1)==tri(:,2) | tri(:,2)==tri(:,3) | tri(:,1)==tri(:,3);
    tri(collapsed, :) = [];
    
    % 2. Remove slivers — raise the area threshold
    e1 = pts(tri(:,2),:) - pts(tri(:,1),:);
    e2 = pts(tri(:,3),:) - pts(tri(:,1),:);
    areas = 0.5 * vecnorm(cross(e1, e2, 2), 2, 2);
    tri(areas < 1e-6, :) = [];   % �? much more aggressive than 1e-10
    
    % Clean up
    [usedIdx, ~, newIdx] = unique(tri(:));
    pts = pts(usedIdx, :);
    tri = reshape(newIdx, size(tri));
    
    % Check
    TR = triangulation(double(tri), double(pts));
    VN = vertexNormal(TR);
    exact_z = abs(VN(:,1)) < 1e-10 & abs(VN(:,2)) < 1e-10 & abs(VN(:,3) - 1) < 1e-10;
    fprintf('%d vertices still have [0 0 1] normals\n', sum(exact_z));

    bad = abs(VN(:,1)) < 1e-10 & abs(VN(:,2)) < 1e-10 & abs(VN(:,3) - 1) < 1e-10;

    % Build adjacency
    edges = [tri(:,[1 2]); tri(:,[2 3]); tri(:,[3 1])];
    A = sparse(edges(:,1), edges(:,2), 1, size(pts,1), size(pts,1));
    A = double((A + A') > 0);
    
    % Replace bad normals with average of neighbors
    badIdx = find(bad);
    for i = 1:numel(badIdx)
        vi = badIdx(i);
        nbrs = find(A(vi, :));
        goodNbrs = nbrs(~bad(nbrs));
        if ~isempty(goodNbrs)
            VN(vi, :) = mean(VN(goodNbrs, :), 1);
            VN(vi, :) = VN(vi, :) / norm(VN(vi, :));
        end
    end
    
    % Verify
    exact_z = abs(VN(:,1)) < 1e-10 & abs(VN(:,2)) < 1e-10 & abs(VN(:,3) - 1) < 1e-10;
    fprintf('%d vertices still have [0 0 1] normals\n', sum(exact_z));

    pts = TR.Points;
    tri = TR.ConnectivityList;

    % 3. Verify
    TR = triangulation(double(tri), double(pts));
    FN = faceNormal(TR);
    FC = (pts(tri(:,1),:) + pts(tri(:,2),:) + pts(tri(:,3),:)) / 3;
    outward = dot(FN, FC - mean(pts,1), 2);
    fprintf('%d / %d faces still point inward\n', sum(outward < 0), size(tri,1));

    % temp_tri = fix_normals(temp_points, temp_tri);

    % clf
    % hold on
    % trisurf(temp_tri, temp_points(:,1), temp_points(:,2), temp_points(:,3), 'Facecolor','red','FaceAlpha',0.2);
    % scatter3(pts_closed(:,1), pts_closed(:,2), pts_closed(:,3), 5, 'filled');

    %% Expand triangulated poonts
    pts = TR.Points;
    tri = TR.ConnectivityList;
    iterations = 2;
    [new_pts, new_tri] = upsample_triangulation(pts, tri, iterations);
    if ~final_triangulation
        writematrix(new_pts, [outdir filesep 'shellpoints_t' num2str(t, '%04d') '.csv']);
    end

    %% Remesh with a poisson algorigthm
    if final_triangulation
        ptCloud = pointCloud(new_pts);

        normals = pcnormals(ptCloud, 18);
        ptCloud = pointCloud(new_pts,'Normal',normals);

        ptCloud = pcdownsample(ptCloud,"gridAverage",20);
        [ptCloud,indices] = removeInvalidPoints(ptCloud);


        [mesh, depth, perVertexDensity] = pc2surfacemesh(ptCloud, "poisson", 6);

        removeDefects(mesh,"duplicate-vertices")
        removeDefects(mesh,"duplicate-faces")
        removeDefects(mesh,"unreferenced-vertices")
        removeDefects(mesh,"degenerate-faces")
        removeDefects(mesh,"nonmanifold-edges")

        % mesh = smoothSurfaceMesh(mesh, 10, 'Method','Taubin', 'ScaleFactor', [0.6 -0.63]);
        % surfaceMeshShow(mesh)

        TF = isWatertight(mesh)

        tri = mesh.Faces;
        VN = mesh.FaceNormals;
        pts = mesh.Vertices;

        %% save the mesh for python

        outfile = fullfile(outdir, sprintf('%04d.obj', t-1));
        fid = fopen(outfile, 'w');

        % vertices — one call, no loop
        fprintf(fid, 'v %f %f %f\n', pts.');       % note the TRANSPOSE

        % normals
        % fprintf(fid, 'vn %f %f %f\n', VN.');
        fprintf(fid, 'vn %f %f %f\n', VN.');
        fprintf(fid, 'f %d//%d %d//%d %d//%d\n', ...
            [tri(:,1) tri(:,1) tri(:,2) tri(:,2) tri(:,3) tri(:,3)].');

        % faces
        fprintf(fid, 'f %d//%d %d//%d %d//%d\n', ...
            [tri(:,1) tri(:,1) tri(:,2) tri(:,2) tri(:,3) tri(:,3)].');

        fclose(fid);
        disp('done')
        toc
    end
end