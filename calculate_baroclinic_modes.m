function [Rd_h, Rd_v, RMS_hv] = calculate_baroclinic_modes(Lat_all, Pres_full, icast, N2c, dz)
% CALCULATE_BAROCLINIC_MODES Computes baroclinic modes for horizontal and vertical velocities.
%
% In MATLAB, the eigenvalue problem is to determine the solution to the equation Av = lambda*v, 
% where A is an n-by-n matrix, v is a column vector of length n, and lambda is a scalar. 
%
% For the horizontal velocity modes the equation to solve is:
%                d/dz[f^2/N^2 dF/dz] = -1/Rd^2 F 
%
% The first step computes the discretised matrix A of the double derivative 
% in the original equation to yield M_h. Then, the eig function [Vh,Dh] = eig(M_h) 
% returns a diagonal matrix Dh of eigenvalues and matrix Vh of right eigenvectors.
%
% Computing the vertical velocity modes follows a similar path except that
% the equation is:
%                1/N^2 d/dz(dG/dz) = -1/(f^2*Rd^2) G 
%
% Both solutions are tested for equivalency via appropriate scaling.

    % =========================================================================
    % 1. CONSTANTS & CORIOLIS PARAMETER
    % =========================================================================
    d2r = pi / 180;
    lat_mean = nanmean(Lat_all(:, icast));
    f = (4 * pi / 86164 * sin(lat_mean * d2r)); 
    g = gsw_grav(Lat_all, Pres_full);

    % =========================================================================
    % 2. HORIZONTAL VELOCITY MODES
    % =========================================================================
    % Note: M_h is the discretized double derivative matrix (loaded externally)
    % run('Matrix_mode_UV_regular_depths.m'); 
    
    % Solve eigenvalue problem for horizontal modes
    [Vh, Dh] = eig(M_h);

    % Sort eigenvalues and compute Deformation Radius (Rd)
    [Lh, indh] = sort(diag(Dh));
    Lh(1) = abs(Lh(1));
    Rd_h = (1 ./ (Lh * f^2)).^0.5;

    % Extract and normalize eigenvectors
    Vh = Vh(:, indh);
    Nmod_h = 10; % Truncation to dominant modes
    mod_h = Vh(:, 1:Nmod_h);
    
    [nzmod, ~] = size(mod_h);
    sgnh = sign(mod_h(1, :));
    sgnh(sgnh == 0) = 1;

    % Kundu/Massimichi Normalization
    summod = (sum(mod_h.^2 * dz) * 1/Pres_full(nzmod)).^0.5;
    modn_h = mod_h ./ summod .* sgnh;

    % =========================================================================
    % 3. VERTICAL VELOCITY MODES
    % =========================================================================
    % Note: Matrices A and B for generalized eigenvalue problem (loaded externally)
    % run('Matrix_mode_W_regular_depths.m');

    % Solve generalized eigenvalue problem A*V = lambda*B*V
    [Vv, Dv] = eig(A, B);

    % Sort eigenvalues and compute Deformation Radius (Rd)
    [Lv, indv] = sort(diag(Dv));
    Rd_v = (1 ./ (f^2 .* Lv)).^0.5;

    % Extract and normalize eigenvectors
    Vv = Vv(:, indv);
    Nmod_v = Nmod_h - 1;
    mod_v = Vv(:, 1:Nmod_v);
    
    sgnv = sign(mod_v(1, :));
    sgnv(sgnv == 0) = 1;
    
    modn_v = zeros(size(mod_v));
    for i = 1:Nmod_v
        summod = sum(mod_v(:, i).^2 .* N2c(:)) * dz * Lv(i);
        summod = sqrt(abs(summod) / Pres_full(nzmod));
        modn_v(:, i) = mod_v(:, i) ./ summod .* sgnv(i);
    end

    % =========================================================================
    % 4. WKB APPROXIMATION & COHERENCY CHECK
    % =========================================================================
    sumNdz = nansum(N2c.^0.5 * dz);
    Rd_WKB = zeros(1, 10);
    for n = 1:10
        Rd_WKB(n) = 1 / (n * pi * f) * sumNdz;
    end

    % Convert to kilometers for operational readability
    Rd1 = Rd_h(2:11) / 1000;
    Rd2 = Rd_v(1:10) / 1000;
    Rd3 = Rd_WKB' / 1000;

    % Root Mean Square differences between solutions
    RMS_hv = sqrt((Rd1 - Rd2).^2);  
    RMS_hwkb = sqrt((Rd1 - Rd3).^2); 
    RMS_vwkb = sqrt((Rd2 - Rd3).^2); 

    % Display structural metrics
    fprintf('RMS between horizontal and vertical modes = %.3f km\n', RMS_hv);
    fprintf('RMS between horizontal modes and WKB      = %.3f km\n', RMS_hwkb);
    fprintf('RMS between vertical modes and WKB        = %.3f km\n', RMS_vwkb);
end
