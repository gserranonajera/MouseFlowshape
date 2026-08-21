function [new_pts, new_tri] = upsample_triangulation(pts, tri, iterations)
    for i = 1:iterations
        [pts, tri] = subdivide(pts, tri);
    end
    new_pts = pts;
    new_tri = tri;
end

function [new_pts, new_tri] = subdivide(pts, tri)
    nt = size(tri, 1);

    % ── Build all edges (each triangle contributes 3) ────────────────────
    edges = [tri(:,[1,2]); tri(:,[2,3]); tri(:,[1,3])];
    edges = sort(edges, 2);                      % canonical order (low,high)
    [uniq_edges, ~, ic] = unique(edges, 'rows'); % deduplicate

    % ── Midpoint for each unique edge ─────────────────────────────────────
    mid_pts = (pts(uniq_edges(:,1),:) + pts(uniq_edges(:,2),:)) / 2;
    new_pts = [pts; mid_pts];

    % ic maps each of the 3*nt edges back to its midpoint row
    % offset by original number of points
    np = size(pts, 1);
    mid_idx = ic + np;                           % global index of midpoint

    % ── Recover per-triangle midpoint indices ─────────────────────────────
    m12 = mid_idx(0*nt + (1:nt));               % midpoint of edge (v1,v2)
    m23 = mid_idx(1*nt + (1:nt));               % midpoint of edge (v2,v3)
    m13 = mid_idx(2*nt + (1:nt));               % midpoint of edge (v1,v3)

    v1 = tri(:,1);  v2 = tri(:,2);  v3 = tri(:,3);

    % ── Each triangle → 4 children ────────────────────────────────────────
    %
    %        v1
    %        /\
    %      m12  m13
    %      / \  / \
    %    v2  m23   v3
    %
    new_tri = [v1,  m12, m13;    % top
               m12,  v2, m23;    % bottom-left
               m13, m23,  v3;    % bottom-right
               m12, m23, m13];   % centre
end