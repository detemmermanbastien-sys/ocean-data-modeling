function [Phi, conv_index] = barotropic_streamfunction_solver(Taux, Tauy, mask, Nx, Ny, dx, dy, R1, iter_max)
% BAROTROPIC_STREAMFUNCTION_SOLVER Calculates the wind-driven ocean circulation.
% Solves the elliptic streamfunction equation (Phi) using a finite difference
% method and an iterative relaxation scheme.
% Explicitly handles complex boundary conditions across all grid edges.

    % Grid step squares and coefficients
    dyc = dy^2;
    dxc = dx^2;
    denom = 2*dxc + 2*dyc;
    num = dxc*dyc;

    Phi_it = zeros(Nx, Ny);
    Phi = zeros(Nx, Ny);
    conv_index = zeros(1, iter_max);

    for it = 2:iter_max
        
        % 1. INTERIOR POINTS CALCULATION (Centered finite differences)
        for i = 2:Nx-1
            for j = 2:Ny-1
                Phi(i,j) = (dyc*(Phi_it(i+1,j) + Phi_it(i-1,j))/denom) ...
                         + (dxc*(Phi_it(i,j+1) + Phi_it(i,j-1))/denom) ...
                         + ((num/denom)*((Tauy(i+1,j) - Tauy(i-1,j))/(2*dx))/R1) ...
                         - ((num/denom)*((Taux(i,j+1) - Taux(i,j-1))/(2*dy))/R1);
            end
        end

        % 2. BOUNDARY CONDITIONS HANDLING
        % Southern Boundary (j = 1)
        j = 1;
        for i = 2:Nx-1
            Phi(i,j) = (dyc*(Phi_it(i+1,j) + Phi_it(i-1,j))/denom) ...
                     + (dxc*(Phi_it(i,j+1) + Phi_it(i,j))/denom) ...
                     + ((num/denom)*((Tauy(i+1,j) - Tauy(i-1,j))/(2*dx))/R1) ...
                     - ((num/denom)*((Taux(i,j+1) - Taux(i,j))/dy)/R1);   
        end

        % Western Boundary (i = 1)
        i = 1;
        for j = 2:Ny-1
            Phi(i,j) = (dyc*(Phi_it(i+1,j) + Phi_it(i,j))/denom) ...
                     + (dxc*(Phi_it(i,j+1) + Phi_it(i,j-1))/denom) ...
                     + ((num/denom)*((Tauy(i+1,j) - Tauy(i,j))/dx)/R1) ...
                     - ((num/denom)*((Taux(i,j+1) - Taux(i,j-1))/(2*dy))/R1);      
        end 
        
        % South-West Corner (i = 1, j = 1)
        i = 1; j = 1;
        Phi(i,j) = (dyc*(Phi_it(i+1,j) + Phi_it(i,j))/denom) ...
                 + (dxc*(Phi_it(i,j+1) + Phi_it(i,j))/denom) ...
                 + ((num/denom)*((Tauy(i+1,j) - Tauy(i,j))/dx)/R1) ...
                 - ((num/denom)*((Taux(i,j+1) - Taux(i,j))/dy)/R1);

        % Northern Boundary (j = Ny)
        j = Ny;
        for i = 2:Nx-1
            Phi(i,j) = (dyc*(Phi_it(i+1,j) + Phi_it(i-1,j))/denom) ...
                     + (dxc*(Phi_it(i,j) + Phi_it(i,j-1))/denom) ...
                     + ((num/denom)*((Tauy(i+1,j) - Tauy(i-1,j))/(2*dx))/R1) ...
                     - ((num/denom)*((Taux(i,j) - Taux(i,j-1))/dy)/R1);
        end
        
        % North-West Corner (i = 1, j = Ny)
        i = 1; j = Ny;
        Phi(i,j) = (dyc*(Phi_it(i+1,j) + Phi_it(i,j))/denom) ...
                 + (dxc*(Phi_it(i,j) + Phi_it(i,j-1))/denom) ...
                 + ((num/denom)*((Tauy(i+1,j) - Tauy(i,j))/dx)/R1) ...
                 - ((num/denom)*((Taux(i,j) - Taux(i,j-1))/dy)/R1);

        % 3. CONVERGENCE CHECK & MASK APPLICATION
        % Evaluates the convergence index and applies the land/ocean mask
        conv_index(it) = max(abs((Phi - Phi_it) ./ Phi) .* mask', [], 'all');
        Phi_it = (Phi + Phi_it) / 2;
        Phi_it = Phi_it .* mask';
    end
end
