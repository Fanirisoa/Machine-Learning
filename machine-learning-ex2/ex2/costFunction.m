function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%


h_theta = sigmoid(transpose(theta)*transpose(X));
int_val = -y.* transpose(log(h_theta)) - (1-y).* transpose(log(1-h_theta));

m = length(y);
J = (sum(int_val))/(m);



% compute the derivative vector
for j = 1:p
    grad(j)  = sum(((h_theta - transpose(y))*X(:, j))/m);
end

% =============================================================

end
