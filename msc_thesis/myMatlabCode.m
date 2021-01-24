%%% Here is the US Census data from 1900 to 2000
%%% Copied from:
%%% https://www.mathworks.com/help/matlab/examples ... 
%%%              /predicting-the-us-population.html
%%% 

%%% Don't use too long lines

% Time interval
t = (1900:10:2000)';

% Population
p = [75.995 91.972 105.711 123.203 131.669 ...
   150.697 179.323 203.212 226.505 249.633 281.422]';

% Plot
plot(t,p,'bo');
axis([1900 2020 0 400]);
title('Population of the U.S. 1900-2000');
ylabel('Millions');

n = length(t);
s = (t-1950)/50;
A = zeros(n);
A(:,end) = 1;
for j = n-1:-1:1
   A(:,j) = s .* A(:,j+1);
end


