%% Clear and Close Figures
clear ; close all; clc;

fprintf('Loading data ...\n');

%% Load Data
data = load('ukay.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f  \n', [X(1:10,:) y(1:10,:)]');

fprintf('\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
%[X mu sigma] = featureNormalize(X);

Xn = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

mu=mean(X)
sigma=std(X)

indices=1:size(X,2);

for i=indices,
    XminusMu=X(:,i)-mu(i);
    Xn(:,i)=XminusMu/sigma(i);
    end


X=Xn;

% Add intercept term to X
X = [ones(m, 1) X];

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Gradient Descent ================
fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 350;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
%[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

J_history = zeros(num_iters, 1);

for iter = 1:num_iters


    hypothesis = X * theta;

    % errors = mx1 column vector
    % y = mx1 column vector
    errors = hypothesis .- y;

    newDecrement = (alpha * (1/m) * errors' * X); 
    
    theta = theta - newDecrement';

    J_history(iter) = computeCostMulti(X, y, theta);
    


end


fprintf('Program paused. Press enter to continue.\n');
pause;

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the sales on Monday (1) 
normalizedDayOfWeek = (1 - mu) / sigma;
normalizedNumberOfItemswithMLPredict = (8 - mu) / sigma;
normalizedNumberOfItemswithIncome = (6 - mu) / sigma;
normalizedNumberOfItemswithBreakEven = (5.5 - mu) / sigma;
normalizedNumberOfItemswithIncomeLoss = (4 - mu) / sigma;

normalizedInputML = [1, normalizedDayOfWeek, normalizedNumberOfItemswithMLPredict];
normalizedInputINC = [1, normalizedDayOfWeek, normalizedNumberOfItemswithIncome];
normalizedInputBE = [1, normalizedDayOfWeek, normalizedNumberOfItemswithBreakEven];
normalizedInputLoss = [1, normalizedDayOfWeek, normalizedNumberOfItemswithIncomeLoss];


MLGradientDescent = normalizedInputML * theta; 
INCGradientDescent = normalizedInputINC * theta; 
BEGradientDescent = normalizedInputBE * theta; 
LossGradientDescent = normalizedInputLoss * theta; 


fprintf(['Predicted Sales on a Monday (using gradient descent): \n']);


 fprintf('ML predicted sales amounts to: %f\n',...
    MLGradientDescent);
fprintf('Sales needed to have income: %f\n',...
    INCGradientDescent);
fprintf('Sales needed to break even: %f\n',...
    BEGradientDescent);
fprintf('Sales consdiered loss: %f\n',...
    LossGradientDescent);
 
 
 






%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% The following code computes the closed form solution for 
% linear regression using the normal equations. 

%% Load Data
data = load('ukay.txt');

X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
%theta = normalEqn(X, y); 
theta=pinv(X'*X)*X'*y;
% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the sales on a Monday with 6 items sold

pricemon = [1, 1, 6] * theta; 

fprintf(['Predicted sales on a Monday (1) with 6 items sold ' ...
         '(using normal equations):\n P%f\n'], pricemon);


% Estimate the sales on a Tuesday with 6 items sold

pricetue = [1, 2, 6] * theta; 

fprintf(['Predicted sales on a Tuesday (2) with 6 items sold ' ...
         '(using normal equations):\n P%f\n'], pricetue);

% Estimate the sales on a Wednesday with 6 items sold

pricewed = [1, 3, 6] * theta; 

fprintf(['Predicted sales on a Wednesday (3) with 6 items sold ' ...
         '(using normal equations):\n P%f\n'], pricewed);

% Estimate the sales on a Thursday with 6 items sold

pricethu = [1, 4, 6] * theta; 

fprintf(['Predicted sales on a Thursday (4) with 6 items sold ' ...
         '(using normal equations):\n P%f\n'], pricethu);


% Estimate the sales on a Friday with 6 items sold

pricefri = [1, 5, 6] * theta; 

fprintf(['Predicted sales on a Friday (5) with 6 items sold ' ...
         '(using normal equations):\n P%f\n'], pricefri);

% Estimate the sales on a Saturday with 6 items sold

pricesat = [1, 6, 6] * theta; 

fprintf(['Predicted sales on a Saturday (6) with 6 items sold ' ...
         '(using normal equations):\n P%f\n'], pricesat);

% Estimate the sales on a Sunday with 6 items sold

pricesun = [1, 7, 6] * theta; 

fprintf(['Predicted sales on a Sunday (7) with 6 items sold ' ...
         '(using normal equations):\n P%f\n'], pricesun);
