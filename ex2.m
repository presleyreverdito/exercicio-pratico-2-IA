%% Inteligencia Artficial - Exercício 2: Regressão Logística

%  Instruções
%  ------------
%
%  Nesta atividade você alterar os seguintes arquivos:
%
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%
%
%  Você não deve alterar o código desta atividade.
%
%

%% Initialização
clear all; close all; clc

%% Carregando os Dados
%  As duas primeiras colunas contém notas de provas de um candidato e a terceira
%  contém o resultado de um processo de seleção (aprovado e reprovado).

data = csvread('ex2data1.txt');
X = data(:, [1, 2]);
y = data(:, 3);

%% =============== Parte 1: Mostrando os dados (plotting) ==================

fprintf('Mostrando os dados + indica y = 1 e o indica y = 0.\n')


plotData(X, y);

%
hold on;
%
xlabel('Nota na prova 1')
ylabel('Nota na prova 2')

%
legend('Aprovado', 'Reprovado')
hold off;

fprintf('Programa pausado. Aperte enter para continuar.\n');
pause;


%% ============ Parte 2: Custo e Gradiente ============

fprintf('Calculando Descida do Gradiente ...\n')

[m, n] = size(X);

% Adciona uma coluna de 1's em x
X = [ones(m, 1) X];

% Valores iniciais dos parametros
initial_theta = zeros(n + 1, 1);

% Calcula e mostra o custo e o gradiente para os valores iniciais dos parametros
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Custo para valores iniciais theta (aprox. 0.693147): %f\n', cost);
fprintf('Gradiente ( aprox. [ -0.100000  -12.009217  -11.262842 ]): \n');
fprintf(' %f \n', grad);

fprintf('Programa pausado. Aperte enter para continuar.\n');
pause;


%% ============= Parte 3: Otimização (fminunc)  =============
%

%  Definições iniciais do fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Chamada para função fminunc
%  A função devolve o parâmetro theta ótimo e o custo neste ponto
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Mostrando o
fprintf('Custo no ponto ótimo encontrado por fminunc (aprox.  0.203506): %f\n', cost);
fprintf('theta (aprox. [ -24.932998  0.204408 0.199618]): \n');
fprintf(' %f \n', theta);

% Mostrando a superfície de decisão
plotDecisionBoundary(theta, X, y);

%
hold on;
%
xlabel('Nota prova 1')
ylabel('Nota prova 2')

% Specified in plot order
legend('Aprovado', 'Reprovado')
hold off;

fprintf('Programa pausado. Aperte enter para continuar.\n');
pause;

%% ============== Parte 4: Prevendo os resultados ==============

fprintf('Prevendo o resultado ...\n')


prob = sigmoid([1 45 85] * theta);
fprintf(['O candidato com notas 45 e 85, será aprovado com  ' ...
         'probabilidade %f\n\n'], prob);

% Calculando a acurária do treinamento
p = predict(theta, X);

fprintf('Accuracia: %f\n', mean(double(p == y)) * 100);

fprintf('Programa pausado. Aperte enter para continuar.\n');
pause;
