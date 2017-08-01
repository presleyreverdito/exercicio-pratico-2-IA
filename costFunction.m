function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Calcula o custo e o gradinete para regressão logística
%   J = COSTFUNCTION(theta, X, y) calcula o custo usando theta como
%   parâmetro para regressão logística e o gradiente da função de custo 
%   considerando os  parâmetros.

% Número de exemplos de treinamento
m = length(y); 

% Você deve preencher as seguintes variáveis:
J = 0;
grad = zeros(size(theta));
media=1/m;
% ====================== COLOQUE SEU CÓDIGO AQUI ======================
% Instruções: Calcule o custo para um determinado theta e preencha a
%             variável J.
%               
%             Calcule as derivadas parciais  com respeito a cada parametro
%             theta e preencha a variável grad.
%             grad deve ter a mesma dimensão de theta  
%
h=sigmoid(X * theta);

J = ( (-y)' * log(h) - (1-y)' * ( log(1-h) ) )/m ;

grad =  (X'*(h-y))/m;


% =============================================================

end
