function [LON, LAT, DATA] = extract_ocean_data(file_nc, var_name, target_date_num)
% EXTRACT_OCEAN_DATA Extracts and temporally aligns Copernicus or MARC NetCDF data.
% Matches the closest temporal slice to a target date and applies validity masks.

    % 1. Read Spatial Coordinates
    lon_c = double(ncread(file_nc, 'longitude')); 
    lat_c = double(ncread(file_nc, 'latitude'));
    [LON, LAT] = meshgrid(lon_c, lat_c);
    
    % 2. Read and Convert Time (Handling different base years 1900/1950/1970)
    info_var = ncinfo(file_nc); 
    var_names = {info_var.Variables.Name};
    
    if ismember('valid_time', var_names)
        time_raw = double(ncread(file_nc, 'valid_time'));
    else
        time_raw = double(ncread(file_nc, 'time'));
    end
    
    t0 = time_raw(1); 
    base_year = 1970; diviseur = 1;
    if t0 > 1e8
        diviseur = 86400; base_year = 1970;
    elseif t0 > 400000
        diviseur = 24; base_year = 1950;
    elseif t0 > 20000 && t0 < 60000
        base_year = 1900; 
    end
    
    t_vec = datenum(base_year, 1, 1) + (time_raw / diviseur);
    
    % 3. Find Closest Temporal Slice
    [~, idx_time] = min(abs(t_vec - target_date_num));
    
    % 4. Extract Data Slice
    info_tmp = ncinfo(file_nc, var_name);
    if length(info_tmp.Dimensions) >= 4
        DATA = squeeze(ncread(file_nc, var_name, [1 1 1 idx_time], [Inf Inf 1 1]));
    else
        DATA = squeeze(ncread(file_nc, var_name, [1 1 idx_time], [Inf Inf 1]));
    end
    
    if size(DATA, 1) == length(lon_c)
        DATA = DATA'; 
    end
end
