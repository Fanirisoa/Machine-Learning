function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% number of features.
p = size(X, 2);
m = length(y);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

transpose_theta = transpose(theta);
h_theta = sigmoid(transpose(theta)*transpose(X));
int_val = -y.* transpose(log(h_theta)) - (1-y).* transpose(log(1-h_theta));

delta = (lambda/(2*m))*(sum(transpose(theta)*theta));

m = length(y);
J = (sum(int_val))/(m) + delta - (lambda/(2*m))*(theta(1)^2) ;

grad(1)  = sum(((h_theta - transpose(y))*X(:, 1))/m);
% compute the derivative vector
for j = 2:p

		grad(j)  = (sum(((h_theta - transpose(y))*X(:, j))/m)) + ((lambda/m)*theta(j));
end

% =============================================================

end
