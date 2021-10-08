clc
close all
clear all
 
X = [ 0 0 1;
      0 1 1;
      1 0 1;
      1 1 1;
    ]
 
D = [ 0
      0
      1
      1
    ]
      
W = 2*rand(1, 3) - 1;
 
for epoch = 1:500
  alpha = 0.9;
 
  dWsum = zeros(3, 1);
   
  N = 4;  
  for k = 1:N
    x = X(k, :)';
    d = D(k);
                        
    v = W*x;
    y = Sigmoid(v);
    %error calculation
    e     = d - y;    
    delta = y*(1-y)*e;
    
    dW = alpha*delta*x;
    
    dWsum = dWsum + dW;
  end
  dWavg = dWsum / N;
  %weight update
  W(1) = W(1) + dWavg(1);
  W(2) = W(2) + dWavg(2);
  W(3) = W(3) + dWavg(3);
 
end
YY=[];
N = 4;
for k = 1:N
  x = X(k, :)';
  v = W*x;
  y = Sigmoid(v);
  YY=[YY y];
end
disp('The prediction Of this network are: ')
disp(YY>.90)

function y = Sigmoid(x)
  y = 1 ./ (1 + exp(-x));
end
