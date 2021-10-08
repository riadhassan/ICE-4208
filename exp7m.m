clc
close all
clear all
randn('seed',3)
rand('seed', 3)
X  = zeros(5, 5, 5);
X(:, :, 1) = [ 0 1 1 0 0;
               0 0 1 0 0;
               0 0 1 0 0;
               0 0 1 0 0;
               0 1 1 1 0
             ];
 
X(:, :, 2) = [ 1 1 1 1 0;
               0 0 0 0 1;
               0 1 1 1 0;
               1 0 0 0 0;
               1 1 1 1 1
             ];
 
X(:, :, 3) = [ 1 1 1 1 0;
               0 0 0 0 1;
               0 1 1 1 0;
               0 0 0 0 1;
               1 1 1 1 0
             ];
 
X(:, :, 4) = [ 0 0 0 1 0;
               0 0 1 1 0;
               0 1 0 1 0;
               1 1 1 1 1;
               0 0 0 1 0
             ];
         
X(:, :, 5) = [ 1 1 1 1 1;
               1 0 0 0 0;
               1 1 1 1 0;
               0 0 0 0 1;
               1 1 1 1 0
             ];
 
D = [ 1 0 0 0 0;
      0 1 0 0 0;
      0 0 1 0 0;
      0 0 0 1 0;
      0 0 0 0 1
    ]
      
W1 = randn(50, 25);
W2 = randn( 5, 50);
 
for epoch = 1:1000         % train
  alpha = 0.9;
  
  N = 5;  
  for k = 1:N
    x = reshape(X(:, :, k), 25, 1);
    d = D(k, :)';
    
    v1 = W1*x;
    y1 = 1 ./ (1 + exp(-v1));
    v  = W2*y1;
    y  = v ./ sum(v);
    e     = d - y;
    delta = e;
 
    e1     = W2'*delta;
    delta1 = y1.*(1-y1).*e1; 
    
    dW1 = alpha*delta1*x';
    W1 = W1 + dW1;
    
    dW2 = alpha*delta*y1';   
    W2 = W2 + dW2;
  end
 
end
yy=[];
N = 5;                        % inference
for k = 1:N
  x  = reshape(X(:, :, k), 25, 1);
  v1 = W1*x;
  y1 = 1 ./ (1 + exp(-v1));
  v  = W2*y1;
  y  = v ./ sum(v);
  yy=[yy y];
end
disp('The prediction Of this network are: ');
disp(yy>0.90)

