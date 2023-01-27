function [phi, phi_x, phi_xx, phi_y, phi_yy, phi_xy, phi_xxxx, phi_yyyy] = ...
    coarse_grain_box(xp, yp, weights, Lx, Ly, sig, xs, ys, rep, calc_ders)
%coarse-graining function that rescales the Gaussian kernel with its area
%within a box 
%kernel function by the its area integral inside the box 
%xp, yp: particle positions
%weights: ones for density, or v_x, or v_y
%Lx, Ly: lengths of box size in x and y
%sig: smoothing scale with a Gaussian
%xs, ys: grid on which to find the coarse-graining 
%rep: number of times to repeat the field on each side
%e.g. rep = 2 gives 25 total boxes 
%calc_ders: true or false, whether to evaluate the derivatives of phi

if nargin<9
  calc_ders = false; %default value for whether to find derivatives
  rep = 1; %default = 1 repetition
elseif nargin<10
  calc_ders = false; %default value for whether to find derivatives
end

%fprintf('rep = %d\n', rep);

Nx = length(xs);
Ny = length(ys);
[xx, yy] = meshgrid(xs, ys);

%number of particles 
Np = length(xp);

%save repeated particles 
xp_rep = zeros((2*rep+1)^2*Np,1);
yp_rep = zeros((2*rep+1)^2*Np,1);
weights_rep = zeros((2*rep+1)^2*Np,1);

count = 0;
for i = -rep:1:rep
    for j = -rep:1:rep
        xp_rep(Np*count+1:Np*(count+1)) = xp + i*Lx;
        yp_rep(Np*count+1:Np*(count+1)) = yp + j*Ly;
        weights_rep(Np*count+1:Np*(count+1)) = weights;
        count = count + 1;
    end
end

phi = zeros(size(xx));
phi_x = NaN(size(xx));
phi_xx = NaN(size(xx));
phi_y = NaN(size(xx));
phi_yy = NaN(size(xx));
phi_xy = NaN(size(xx));
phi_xxxx = NaN(size(xx));
phi_yyyy = NaN(size(xx));

%cdf of the Gaussian function 
F = @(x) 0.5*(1 + erf(x/(sqrt(2)*sig)));

for i = 1:Ny
    %fprintf('%i/%i\n', i, Ny)
    for j = 1:Nx
        dist_squared = (xp_rep - xx(i,j)).^2 + (yp_rep - yy(i,j)).^2;
        integral = (F(Lx-xx(i,j)) - F(-xx(i,j)))*(F(Ly-yy(i,j)) - F(-yy(i,j)));
        gauss = exp(-dist_squared/(2*sig^2))/(2*pi*sig^2);
        phi(i,j) = sum(weights_rep .* gauss)/integral;
        if calc_ders
            xc = xp_rep - xx(i,j);
            yc = yp_rep - yy(i,j);
            phi_x(i,j)  =   sum(weights_rep .* gauss .* xc ./ sig^2);
            phi_y(i,j)  =   sum(weights_rep .* gauss .* yc ./ sig^2);
            phi_xx(i,j) =   sum(weights_rep .* gauss .* (xc.^2 - sig^2) ./ sig^4);
            phi_yy(i,j) =   sum(weights_rep .* gauss .* (yc.^2 - sig^2) ./ sig^4);
            phi_xy(i,j) =   sum(weights_rep .* gauss .* xc .* yc ./ sig^4);
            phi_xxxx(i,j) = sum(weights_rep .* gauss .* (xc.^4 - 6*(sig^2).*xc.^2 + 3*(sig^4)) ./ sig^8);
            phi_yyyy(i,j) = sum(weights_rep .* gauss .* (yc.^4 - 6*(sig^2).*yc.^2 + 3*(sig^4)) ./ sig^8);
        end
    end
end


end