function export(xx, yy)
    [m n] = size(xx);
    rev = zeros([m n]);
    
    for i = 1:m
        mat = reshape(xx(i,:), [20, 20])';

        rev(i,:) = reshape(mat, [1 n]);

    end

    yy(yy == 10) = 0;

    csvwrite("ex3data1.txt", [rev yy]);
end
