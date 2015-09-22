%% =============    === Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc;

fprintf('Loading data ...\n');

%% Load Data
X=[load('ielts_band.txt'),load('fsw_points.txt'),load('job_experience_years.txt'),load('fee_encashed_date.txt'),load('PER_recvd_date.txt')];
y=load('visa_issued_date.txt');
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f %.0f %.0f %.0f], y = %.0f  \n', [X(1:10,:) y(1:10,:)]');

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
num_iters = 600;

% Init Theta and Run Gradient Descent 
theta = zeros(6, 1);
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
    %J_history(iter) = 1/(2*m)*(sum((X*theta).-y).^2);


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

% Estimate the length of visa approval of a 6 ielts band ,FSW points of 71 ,job experience of 10 years ,fee encashed 111 days and PER issued 122 days
normalizedIELTSBAND1 = (6 - mu) / sigma;
normalizedFSWSCORE1 = (71 - mu) / sigma;
normalizedjobexperience1 = (10 - mu) / sigma;
normalizedfeeEncashed1= (111 - mu) / sigma;
normalizedPERissuedate1= (122 - mu) / sigma;

% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.


normalizedInput1 = [1,normalizedIELTSBAND1, normalizedFSWSCORE1,normalizedjobexperience1,normalizedfeeEncashed1,normalizedPERissuedate1 ];
Predict_GradDescent1 = normalizedInput1 * theta; % You should change this

fprintf(['Predicted visa approval of applicant w/ 6 IELTS score , 71 Federal Skilled Worker (FSW) points with job experience of 10 years ,fee encashed 111 days and PER issued 122 days ' ...
         '(using gradient descent):\n %d days\n'], Predict_GradDescent1);

% Estimate the length of visa approval of a 7.5 ielts band ,FSW points of 74 ,job experience of 4 years ,fee encashed 76 days and PER issued 105 days
normalizedIELTSBAND1 = (7.5 - mu) / sigma;
normalizedFSWSCORE1 = (74 - mu) / sigma;
normalizedjobexperience1 = (4 - mu) / sigma;
normalizedfeeEncashed1= (76 - mu) / sigma;
normalizedPERissuedate1= (105 - mu) / sigma;

% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.


normalizedInput1 = [1,normalizedIELTSBAND1, normalizedFSWSCORE1,normalizedjobexperience1,normalizedfeeEncashed1,normalizedPERissuedate1 ];
Predict_GradDescent1 = normalizedInput1 * theta; % You should change this

fprintf(['Predicted visa approval of applicant w/ 7.5 IELTS score , 74 Federal Skilled Worker (FSW) points with job experience of 4 years ,fee encashed 76 days and PER issued 105 days ' ...
         '(using gradient descent):\n %d days\n'], Predict_GradDescent1);

% Estimate the length of visa approval of a 6 ielts band ,FSW points of 65 ,job experience of 3 years ,fee encashed 125 days and PER issued 156 days
normalizedIELTSBAND1 = (6 - mu) / sigma;
normalizedFSWSCORE1 = (65 - mu) / sigma;
normalizedjobexperience1 = (3 - mu) / sigma;
normalizedfeeEncashed1= (125 - mu) / sigma;
normalizedPERissuedate1= (156 - mu) / sigma;


normalizedInput1 = [1,normalizedIELTSBAND1, normalizedFSWSCORE1,normalizedjobexperience1,normalizedfeeEncashed1,normalizedPERissuedate1 ];
Predict_GradDescent1 = normalizedInput1 * theta; % You should change this

fprintf(['Predicted visa approval of applicant w/ 6 IELTS score , 65 Federal Skilled Worker (FSW) points with job experience of 3 years ,fee encashed 125 days and PER issued 156 days ' ...
         '(using gradient descent):\n %d days\n'], Predict_GradDescent1);




%% ================ Part 3: Normal Equations ================
fprintf('\n'); 
fprintf('Solving with normal equations...\n');

% The following code computes the closed form solution for 
% linear regression using the normal equations. 

%% Load Data
X=[load('ielts_band.txt'),load('fsw_points.txt'),load('job_experience_years.txt'),load('fee_encashed_date.txt'),load('PER_recvd_date.txt');];
y=load('visa_issued_date.txt');
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

% Estimate the length of approval of a 6 ielts band ,FSW points of 71, 10 years' job experience, fee encashed 111 days, and PER issued 122 days
predict_normalEqn = [1, 6, 71,10,111,122] * theta; 

fprintf(['Predicted visa issuane of applicant w/ IELTS score of 6 , 71 Federal Skilled Workerd (FSW) points ,job experience of 10 years ,fee encashed 111 days and PER issued 122 days' ...
         '(using normal equations):\n %d days\n'], predict_normalEqn);


% Estimate the length of approval of a 7.5 ielts band ,FSW points of 74, 4 years' job experience, fee encashed 76 days, and PER issued 105 days
predict_normalEqn = [1, 7.5, 74,4,76,105] * theta; 

fprintf(['Predicted visa issuane of applicant w/ IELTS score of 7.5 , 74 Federal Skilled Workerd (FSW) points ,job experience of 4 years ,fee encashed 76 days and PER issued 105 days' ...
         '(using normal equations):\n %d days\n'], predict_normalEqn);

% Estimate the length of approval of a 6 ielts band ,FSW points of 65, 3 years' job experience, fee encashed 125 days, and PER issued 156 days
predict_normalEqn = [1, 6, 65,3,125,156] * theta; 

fprintf(['Predicted visa issuane of applicant w/ IELTS score of 6 , 65 Federal Skilled Workerd (FSW) points ,job experience of 3 years ,fee encashed 125 days and PER issued 156 days' ...
         '(using normal equations):\n %d days\n'], predict_normalEqn);