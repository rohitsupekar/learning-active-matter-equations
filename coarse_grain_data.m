%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% This script loads particle data and performs coarse-graining %%%%%%%
%%%%%% Coarse-grained data for density and velocity fields at saved %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
addpath('src/');

%load particle data
load('data/quinke_particle_data_S2.mat');

%coarse-graining width
sigma = 0.045; %in mm units

%number of grid points in x, y directions
Nx = 100;
Ny = 50;

%make video?
makeVideo = false;

%set number of cores to use for performing the coarse-graining of different
%frames in parallel
%num_cores = 2; 

%unit conversions: pixels --> mm, frame number --> seconds
%values below adjusted so mean(rho)*pi*D^2/4 = 0.11 and mean(vx) = 1.2 mm/s
%so as to match the values reported in Geyer et. al, 2018
%this is only so the video coordinates from the SI movie match the data 
mm_per_pixel =  10^(-3)*(330/350)*0.9038;
sec_per_frame = 1.2028/500; 

%set to 0 to ignore periodicity
rep=0; 

save_path = 'data/';

if ~exist(save_path, 'dir')
       mkdir(save_path);
end

%parpool('local', num_cores)

%% find box size 
Nt = length(AllData)-1; %remove last element because its empty
t = 0:sec_per_frame:(Nt-1)*sec_per_frame;

max_x_ = zeros(Nt, 1);
min_x_ = zeros(Nt, 1);
max_y_ = zeros(Nt, 1);
min_x_ = zeros(Nt, 1);

for i=1:Nt
    max_x_(i) = max(AllData(i).xy_withvel_smooth(:, 1));
    min_x_(i) = min(AllData(i).xy_withvel_smooth(:, 1));
    
    max_y_(i) = max(AllData(i).xy_withvel_smooth(:, 2));
    min_y_(i) = min(AllData(i).xy_withvel_smooth(:, 2));
end

max_x = max(max_x_);
min_x = min(min_x_);

max_y = max(max_y_);
min_y = min(min_y_);
%% define grid of Nx, Ny points from min_x to max_x, min_y to max_y

x = (linspace(min_x, max_x, Nx) - min_x)*mm_per_pixel;
y = (linspace(min_y, max_y, Ny) - min_y)*mm_per_pixel;

Lx = x(end) - x(1);
Ly = y(end) - y(1);

%% coarse-graining 
rho = zeros(Ny, Nx, Nt);
%mass fluxes
Fx = zeros(Ny, Nx, Nt);
Fy = zeros(Ny, Nx, Nt);

for i=1:Nt
    fprintf('%s: Time point: %0.4d/%0.4d\n', datestr(now), i, Nt)
    
    xp = (AllData(i).xy_withvel_smooth(:,1) - min_x)*mm_per_pixel;
    yp = (AllData(i).xy_withvel_smooth(:,2) - min_y)*mm_per_pixel;
    
    vxp = AllData(i).vx_vy_smooth(:, 1)*mm_per_pixel/sec_per_frame;
    vyp = AllData(i).vx_vy_smooth(:, 2)*mm_per_pixel/sec_per_frame;

    rho(:, :, i) = coarse_grain_box(xp, yp, 1, Lx, Ly, sigma, x, y, rep, false);
    Fx(:, :, i) = coarse_grain_box(xp, yp, vxp, Lx, Ly, sigma, x, y, rep, false);
    Fy(:, :, i) = coarse_grain_box(xp, yp, vyp, Lx, Ly, sigma, x, y, rep, false);

end

%% get velocity fields 
vx = Fx./rho;
vy = Fy./rho;

%% save data
save(sprintf('%s/coarse_grained_field(y,x,t).mat', save_path), 'rho', 'vx', 'vy', 't', 'x', 'y', '-v6');

%% figures
ind = 10;
%particle positions 
xp = (AllData(i).xy_withvel_smooth(:,1) - min_x)*mm_per_pixel;
yp = (AllData(i).xy_withvel_smooth(:,2) - min_y)*mm_per_pixel;

rho_max = max(rho(:));
rho_min = min(rho(:));

vx_max = max(vx(:));
vx_min = min(vx(:));

vy_max = max(vy(:));
vy_min = min(vy(:));

figure('position', [100, 100, 700, 600]); 
ax1 = subplot(311); hold on;
p1=pcolor(x, y, rho(:, :, ind)); shading interp;
axis equal; colorbar; colormap jet;
title(sprintf('Density [# particles/unit area](frame = %i)', ind))
caxis([rho_min, rho_max])
scatter(xp, yp, 5, 'k', 'filled')
xlabel('x [mm]'); ylabel('y [mm]');
xlim([min(x), max(x)]);
ylim([min(y), max(y)]);

ax2 = subplot(312); box on;
p2=pcolor(x, y, vx(:, :, ind)); shading interp;
axis equal; colorbar; colormap jet;
caxis([vx_min, vx_max]);
title('Horizontal velocity [mm/s]')
xlabel('x [mm]'); ylabel('y [mm]');
xlim([min(x), max(x)]);
ylim([min(y), max(y)]);

ax3 = subplot(313); box on;
p3=pcolor(x, y, vy(:, :, ind)); shading interp;
axis equal; colorbar;
title('Vertical velocity [mm/s]')
xlabel('x [mm]'); ylabel('y [mm]');
caxis([vy_min, vy_max]);
xlim([min(x), max(x)]);
ylim([min(y), max(y)]);

print('data/coarse_grained_frame', '-dpng', '-r200')

%% make movie with all the fields 

if makeVideo
    fps = 20;
    video_name = 'data/coarse_fields_movie';

    rho_max = max(rho(:));
    rho_min = min(rho(:));

    vx_max = max(vx(:));
    vx_min = min(vx(:));

    vy_max = max(vy(:));
    vy_min = min(vy(:));

    figure('position', [100, 100, 600, 700]);
    ax1 = subplot(311);
    p1=pcolor(x, y, rho(:, :, 1)); shading interp;
    axis equal; colorbar; colormap jet;
    caxis([rho_min, rho_max])
    xlim([min(x), max(x)]);
    ylim([min(y), max(y)]);

    ax2 = subplot(312);
    p2=pcolor(x, y, vx(:, :, 1)); shading interp;
    axis equal; colorbar; colormap jet;
    caxis([vx_min, vx_max]);
    xlim([min(x), max(x)]);
    ylim([min(y), max(y)]);

    ax3 = subplot(313);
    p3=pcolor(x, y, vy(:, :, 1)); shading interp;
    axis equal; colorbar;
    caxis([vy_min, vy_max]);
    xlim([min(x), max(x)]);
    ylim([min(y), max(y)]);

    if makeVideo == true
        v = VideoWriter(video_name, 'MPEG-4');
        v.FrameRate = fps;
        open(v);

        for i=1:Nt
           i
           set(p1, 'CData', rho(:, :, i)); 
           ax1.Title.String = sprintf('Density (frame = %i)', i);

           set(p2, 'CData', vx(:, :, i)); 
           ax2.Title.String = sprintf('v_x (frame = %i)', i);

           set(p3, 'CData', vy(:, :, i)); 
           ax3.Title.String = sprintf('v_y (frame = %i)', i);

           frame = getframe(gcf);
           writeVideo(v,frame);
        end

        close(v);

    end
    
end

%%
%delete(gcp('nocreate'))



















