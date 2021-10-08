%Take sigmoid function matlab file inside the current directory
clc
close all
clear all
%Input           
X = [ 0 0 1;
      0 1 1;
      1 0 1;
      1 1 1;
    ]
%Target output
D = [ 0
      0
      1
      1
    ]
%Random weight generation      
W = 2*rand(1, 3) - 1;
 
for epoch = 1:100           % train
  alpha = 0.9;
  
  N = 4;  
  for k = 1:N
    x = X(k, :)';
    d = D(k);
 % Network Layer
    v = W*x
    y = Sigmoid(v)
    
    e     = d - y;  
    delta = y*(1-y)*e;
  
    dW = alpha*delta*x;    % delta rule    
    
    W(1) = W(1) + dW(1); 
    W(2) = W(2) + dW(2);
    W(3) = W(3) + dW(3);    
  end
 
end
YY=[];
N = 4;                        % inference
for k = 1:N
  x = X(k, :)';
  v = W*x;
  y = Sigmoid(v);
  YY=[YY y];
end
disp('The prediction Of this network are: ')
disp(YY>.90)

function y = Sigmoid(x)
  y = 1 / (1 + exp(-x));
end

