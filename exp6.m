clc
close all
clear all
randn('seed',3)
rand('seed',3)         

X = [ 0 0;
      0 1;
      1 0;
      1 1;
    ];

D = [ 0
      1
      1
      0
    ];
    
W1 = -1 +2*rand(4,2);
W2 = -1 +2*rand(1,4);
 
for epoch = 1:2000           % train
  alpha = 0.9;
  
  N = 4;  
  for k = 1:N
    x = X(k, :)';
    d = D(k);
    
    v1 = W1*x;
    y1 = 1 ./ (1 + exp(-v1));
    v  = W2*y1;
    y = 1 / (1 + exp(-v));
    
    e     = d - y;
    delta = y*(1-y)*e;
 
    e1     = W2'*delta;
    delta1 = y1.*(1-y1).*e1; 
    
    dW1 = alpha*delta1*x';
    W1  = W1 + dW1;
    
    dW2 = alpha*delta*y1';    
    W2  = W2 + dW2;
  end
 
end
yy=[];
N = 4;                        % inference
for k = 1:N
  x  = X(k, :)';
  v1 = W1*x;
  y1 = 1 ./ (1 + exp(-v1));;
  v  = W2*y1;
  y  = 1 ./ (1 + exp(-v));
  yy=[yy y];
end
disp('The prediction Of this network are: ');
disp(yy>0.90);

