clear;
clc;
%Input Patterns
E=[1 1 1 1 1 -1 -1 -1 1 1 1 1 1 -1 -1 -1 1 1 1 1];
F=[1 1 1 1 1 -1 -1 -1 1 1 1 1 1 -1 -1 -1 1 -1 -1 -1];
x(1,1:20)=E;
x(2,1:20)=F;
w(1:20)=0;
t=[1 -1];
b=0;
for i=1:2
    w=w+x(i,1:20)*t(i);
    b=b+t(i);
end
disp('Weight matrix');
disp(w);
disp('Bias');
disp(b);

