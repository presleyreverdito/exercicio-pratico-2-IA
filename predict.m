function p = predict(theta, X)
%PREDICT Preve se a classe é 0 ou 1 usando regressão logística

%   p = PREDICT(theta, X) calcula as previsões de X usando um limiar
%   (threshold) em 0.5 (i.e., se sigmoid(theta'*x) >= 0.5, preve a classe 1)


m = size(X, 1); % Numero de amostras de treinamento

% Você deve preencher a variável p
p = zeros(m, 1);

% ====================== COLOQUE SEU CÓDIGO AQUI ======================
p = sigmoid(X*theta)>=0.5;


% =========================================================================


end
