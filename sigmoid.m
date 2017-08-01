function g = sigmoid(z)
%SIGMOID Calcula a função sigmoid
%   J = SIGMOID(z) calcula o sigmoid de z.

%  Você deve preencher a variável g
g = zeros(size(z));

% ====================== COLOQUE SEU CÓDIGO AQUI ======================
g= 1./(1+exp(-z));


% =============================================================

end
