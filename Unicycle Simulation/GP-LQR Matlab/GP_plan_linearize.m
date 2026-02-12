function [A, B] = GP_plan_linearize(x, u, dt)
%     theta = x(3);
%     v = x(4);
    
    A = [...
        1, 0, -x(4)*sin(x(3))*dt, cos(x(3))*dt;
        0, 1,  x(4)*cos(x(3))*dt, sin(x(3))*dt;
        0, 0, 1,                0;
        0, 0, 0,                1];
    
    B = [...
        0, 0;
        0, 0;
        dt, 0;
        0, dt];
end