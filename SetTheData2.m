% Matlab code, which organizes the data,
% downloads the dark measurement and normalizes according to the absorption graph.
function M = SetTheData(Matrix)

% If the number of empty lines is 5, it does not include 50&200.
% And if the number of lines is 7, then it is inclusive.
c = find(isnan(Matrix(:,1)));


meanDark = mean(Matrix(c(1)+1:c(2)-1,7:262),1);  % mean Dark calculate

con0ppmmean = mean(meanDark - Matrix(c(2)+1:c(3)-1,7:262),1);   % mean 0 ppm calculate 

con100ppm = meanDark - Matrix(c(3)+1:c(4)-1,7:262); % Dark - Matrix of 100ppm 
M{1} = con100ppm(:,:)./con0ppmmean;   % Normelaized the 100ppm with the 0ppm

con300ppm = meanDark - Matrix(c(4)+1:c(5)-1,7:262); % Dark - Matrix of 300ppm 
M{2} = con300ppm(:,:)./con0ppmmean;   % Normelaized

if (length(c) == 5)     % checking if the Exel including 50&200 ppm or not
                 % not including 50&200ppm
    con500ppm = meanDark - Matrix(c(5)+1:end,7:262); % Dark - Matrix of 500ppm 
    M{3} = con500ppm(:,:)./con0ppmmean;  % Normelaized

                 % including 50&200ppm
elseif (length(c) == 7)
    con500ppm = meanDark - Matrix(c(5)+1:c(6)-1,7:262); % Dark - Matrix of 500ppm 
    M{3} = con500ppm(:,:)./con0ppmmean;   % Normelaized

    con50ppm = meanDark - Matrix(c(6)+1:c(7)-1,7:262); % Dark - Matrix of 50ppm 
    M{4} = con50ppm(:,:)./con0ppmmean;    % Normelaized

    con200ppm = meanDark - Matrix(c(7)+1:end,7:262); % Dark - Matrix of 200ppm 
    M{5} = con200ppm(:,:)./con0ppmmean;   % Normelaized
end
end
