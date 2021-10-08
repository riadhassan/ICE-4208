clc
clear all
close all
 
X=[2,3]
W1=ones(2,2);
W2=ones(2,2);
Y = feedforward2(X,W1,W2)
 
function [ Y ] = feedforward2( X,W1,W2)
X=transpose(X);          %X needs to be a column vector
A = sigmf(W1*X,[1 0])     %Values of the first hidden layer  
Y = sigmf(W2*A,[1 0])    %Output Values of the network
Y = transpose(Y);     
end
