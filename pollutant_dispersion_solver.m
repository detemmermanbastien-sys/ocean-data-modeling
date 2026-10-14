function [C, x_traj, y_traj] = pollutant_dispersion_solver(ub, vb, Nx, Ny, dx, dy, days, start_pos)
% POLLUTANT_DISPERSION_SOLVER Simulates pollutant transport using Eulerian and Lagrangian methods.
% Implements an Arakawa-C grid upwind scheme for continuous concentration fields (e.g., oil spills)
% and a 4th-order Runge-Kutta (RK4) integration for discrete particle tracking.

    % =========================================================================
    % 1. ARAKAWA-C GRID INITIALIZATION
    % =========================================================================
    % Staggering velocities for numerical stability
    un = zeros(Nx+1, Ny);
    vn = zeros(Nx, Ny+1);
    
    un(1,:) = ub(1,:); un(Nx+1,:) = ub(Nx,:);
    vn(:,1) = vb(:,1); vn(:,Ny+1) = vb(:,Ny);
    
    for i = 2:Nx
        for j = 2:Ny
            un(i,j) = (ub(i-1,j) + ub(i,j)) / 2;
            vn(i,j) = (vb(i,j-1) + vb(i,j)) / 2;
        end
    end

    % Dynamic Time Step Calculation (CFL Condition)
    dt_cfl = floor(min(dx / max(abs(un(:))), dy / max(abs(vn(:)))));
    Nt = floor(days * 24 * 3600 / dt_cfl);

    % =========================================================================
    % 2. EULERIAN ADVECTION (UPWIND SCHEME FOR CONCENTRATION FIELD)
    % =========================================================================
    C = zeros(Nx, Ny, Nt);
    C(220:230, 40:50, 1) = 500; % Initial pollutant patch injection
    
    Fx_am = zeros(Nx, Ny); Fx_av = zeros(Nx, Ny);
    Fy_am = zeros(Nx, Ny); Fy_av = zeros(Nx, Ny);

    for t = 2:Nt
        for i = 2:Nx-1
            for j = 2:Ny-1
                % Upwind flux computations to prevent numerical oscillations
                if un(i,j) > 0, Fx_am(i,j) = un(i,j) * C(i-1,j,t-1);
                else, Fx_am(i,j) = un(i,j) * C(i,j,t-1); end
                
                if un(i+1,j) > 0, Fx_av(i,j) = un(i+1,j) * C(i,j,t-1);
                else, Fx_av(i,j) = un(i+1,j) * C(i+1,j,t-1); end

                if vn(i,j) > 0, Fy_am(i,j) = vn(i,j) * C(i,j-1,t-1);
                else, Fy_am(i,j) = vn(i,j) * C(i,j,t-1); end
                   
                if vn(i,j+1) > 0, Fy_av(i,j) = vn(i,j+1) * C(i,j,t-1);
                else, Fy_av(i,j) = vn(i,j+1) * C(i,j+1,t-1); end
            end
        end
        % Mass conservation update
        C(:,:,t) = C(:,:,t-1) + dt_cfl * (((Fx_am - Fx_av)/dx) + ((Fy_am - Fy_av)/dy));
    end

    % =========================================================================
    % 3. LAGRANGIAN PARTICLE TRACKING (RUNGE-KUTTA 4th ORDER)
    % =========================================================================
    x_traj = zeros(1, Nt); y_traj = zeros(1, Nt);
    x_traj(1) = start_pos(1); y_traj(1) = start_pos(2);
    
    for t = 2:Nt
        % RK4 integration steps for precise advection
        xk1 = dt_cfl * interp2(ub', x_traj(t-1), y_traj(t-1)) / dx;
        yk1 = dt_cfl * interp2(vb', x_traj(t-1), y_traj(t-1)) / dy;
        
        xk2 = dt_cfl * interp2(ub', x_traj(t-1) + (xk1/2), y_traj(t-1) + (yk1/2)) / dx;
        yk2 = dt_cfl * interp2(vb', x_traj(t-1) + (xk1/2), y_traj(t-1) + (yk1/2)) / dy;
        
        xk3 = dt_cfl * interp2(ub', x_traj(t-1) + (xk2/2), y_traj(t-1) + (yk2/2)) / dx;
        yk3 = dt_cfl * interp2(vb', x_traj(t-1) + (xk2/2), y_traj(t-1) + (yk2/2)) / dy;
        
        xk4 = dt_cfl * interp2(ub', x_traj(t-1) + xk3, y_traj(t-1) + yk3) / dx;
        yk4 = dt_cfl * interp2(vb', x_traj(t-1) + xk3, y_traj(t-1) + yk3) / dy;

        x_traj(t) = x_traj(t-1) + (1/6) * (xk1 + 2*xk2 + 2*xk3 + xk4);
        y_traj(t) = y_traj(t-1) + (1/6) * (yk1 + 2*yk2 + 2*yk3 + yk4);
    end
end
