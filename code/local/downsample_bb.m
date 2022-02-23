% Downsample bigbrain
% From https://bigbrainwarp.readthedocs.io/en/latest/pages/tutorial_gradients.html
addpath(genpath('~/Desktop/micaopen'));
hems = {'L', 'R'};
for hem_idx = 1:2
    hem = hems{hem_idx};
    pial_surf_path = sprintf('~/Desktop/bigbrain/BBW_BigData/spaces/tpl-bigbrain/tpl-bigbrain_hemi-%s_desc-pial.obj', hem);
    disp(pial_surf_path)
    BB = SurfStatAvSurf(pial_surf_path);
    % set the target number of faces to 20480 corresponding to ico5
    % see https://brainder.org/2016/05/31/downsampling-decimating-a-brain-surface/
    numFaces= 20480;
    patchNormal = patch('Faces', BB.tri, 'Vertices', BB.coord.','Visible','off');
    Sds = reducepatch(patchNormal,numFaces);
    [~, bb_downsample]  = intersect(patchNormal.Vertices,Sds.vertices,'rows');
    BB10.tri = double(Sds.faces);
    BB10.coord   = Sds.vertices';
    
    % For each vertex on BB, find nearest neighbour on BB10, via mesh neighbours
    nn_bb = zeros(1,length(BB.coord));
    edg = SurfStatEdg(BB);
    parfor ii = 1:length(BB.coord)
            if mod(ii, 1000) == 0
                disp(ii)
            end
            nei = unique(edg(sum(edg==ii,2)==1,:));
            if isempty(nei) && ismember(ii,bb_downsample)
                    nn_bb(ii) = ii;
            else
                    while sum(ismember(nei, bb_downsample))==0
                    nei = [unique(edg(sum(ismember(edg,nei),2)==1,:)); nei];
                    end
            end
            matched_vert = nei(ismember(nei, bb_downsample));
            if length(matched_vert)>1  % choose the mesh neighbour that is closest in Euclidean space
                    n1 = length(matched_vert);
                    d = sqrt(sum((repmat(BB.coord(1:3,ii),1,n1) - BB.coord(:,matched_vert)).^2));
                    [~, idx] = min(d);
                    nn_bb(ii) = matched_vert(idx);
            else
                    nn_bb(ii) = matched_vert;
            end
    end
    outname = sprintf('tpl-bigbrain_hemi-%s_desc-pial_downsampled', hem);
    save(outname, "BB10", "bb_downsample", "nn_bb")
end