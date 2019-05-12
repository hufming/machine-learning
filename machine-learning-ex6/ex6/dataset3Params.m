function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
v = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
n = size(v,1)
c_sigma_pair = zeros(n*n,2);
predict_error = zeros(n*n,1);

for c = 1:n
  for s = 1:n
    c_sigma_pair(n*(c-1)+s,1) = v(c);
    c_sigma_pair(n*(c-1)+s,2) = v(s);
  endfor
endfor

for i=1:n*n
  C = c_sigma_pair(i,1);
  sigma =   c_sigma_pair(i,2);
%  fprintf('in Loop ==== C = %f;sigma =  %f ===.\n',C,sigma);
  
  model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2,sigma)); 
  predictions = svmPredict(model, Xval);
  predict_error(i) =  mean(double(predictions ~= yval));
%  fprintf('in Loop predict_error =  %f .\n',predict_error(i));
endfor

[v,idx] = min(predict_error);
C = c_sigma_pair(idx,1);
sigma = c_sigma_pair(idx,2);
fprintf('Result : C = %f;sigma =  %f;mean  = %f.\n',C,sigma,predict_error(idx));

% =========================================================================

end
