function [  ] = see_data(  )

folder = '../datas/S1L2/';

cctotal = load([folder 'CCtotal.mat']); cctotal = cctotal.CCtotal;
rsp_tpf = load([folder 'Rsp_tPointsFit.mat']); rsp_tpf = rsp_tpf.Rsp_tPointsFit;
celllist = load([folder 'targetCellListANOVA.mat']); celllist = celllist.targetCellListANOVA;
coors = parse_coor(cctotal);

im = zeros(512,512);
for j = 1:length(coors)
    coor = coors{j};
    rsp = rsp_tpf(j);
    for i = 1:size(coor,1)
        im(coor(i,1),coor(i,2)) = rsp;
    end
end
imagesc(im); axis equal;

end

