clc
clearvars

syms q1 q2 q3 q4

L1 = .5;
L2 = .5;
L3 = 1;


fk_endeff_symbolic = [
          L1*cos(q2) + L2*cos(q2 + q3) + L3*cos(q2 + q3 + q4), ...
     q1 + L1*sin(q2) + L2*sin(q2 + q3) + L3*sin(q2 + q3 + q4)
     ]
 
    

Jacobian_symbolic = simplify(jacobian(fk_endeff_symbolic, [q1 q2 q3 q4]))