function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

display(length(computeCost(X, y, theta)))

J_history(1)= computeCost(X, y, theta);

% number of features.
p = size(X, 2);
m = length(y);

deriv = [0;0]
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    % create a copy of theta for simultaneous update.
    theta_prev = theta;

    % compute the derivative vector
    h_theta = transpose(theta)*transpose(X);

     % simultaneous update theta using theta_prev.
    for j = 1:p

        deriv(j)  = ((h_theta - transpose(y))*X(:, j))/m;

    end

    % update theta_j
    theta = theta_prev-(alpha*deriv);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

   
end

end
