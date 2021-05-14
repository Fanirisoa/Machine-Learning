function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
m = length(y);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


fprintf(['size(theta): '], size(theta));



transpose_theta = transpose(theta);
h_theta = sigmoid(transpose(theta)*transpose(X));
inter_val = ((h_theta - transpose(y))*(h_theta - transpose(y)))/(2*m)
delta = (lambda/(2*m))*(sum(transpose(theta)*theta));
J = sum(inter_val) + delta;










% =========================================================================

grad = grad(:);

end
