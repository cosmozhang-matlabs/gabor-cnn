function [ coor ] = parse_coor( input_vector )

% coor: n*2 matrix, each row is [x,y]

if iscell(input_vector)
    ncell = length(input_vector);
    coor = cell(1,ncell);
    for i = 1:ncell
        coor{i} = parse_coor(input_vector{i});
    end
else

    imw = 512;
    imh = 512;

    npts = length(input_vector);
    input_vector = reshape(input_vector,[npts,1]);
    coor = zeros(npts,2);
    coor(:,1) = floor((input_vector-1)./imw)+1;
    coor(:,2) = input_vector - (coor(:,1)-1).*imw;

end

end

