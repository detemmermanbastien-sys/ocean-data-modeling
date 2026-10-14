function T_Master = glider_multisensor_fusion(dir_master, dir_g1, dir_g2)
% GLIDER_MULTISENSOR_FUSION Data engineering pipeline for autonomous gliders.
% Parses, cleans, and synchronizes raw multi-sensor data (CTD, UVP6, ADCP, UBAT).
%
% Features:
% - Raw text parsing and regex time-synchronization for .pld1/.raw files.
% - Spatial/Temporal data grafting using K-Nearest Neighbors (KNN) and Haversine distance.
% - Bio-optical signal processing (Messié et al. 2019) for plankton discrimination.
% - TEOS-10 thermodynamic density calculations (GSW toolbox).
%
% Usage:
% T_Master = glider_multisensor_fusion('path/to/sea090', 'path/to/g1', 'path/to/g2');

    % --- PARAMETERS ---
    DIST_MAX_KM = 5;
    DLIM_Z = 5;
    
    if isempty(which('gsw_sigma0'))
        warning('Gibbs SeaWater (GSW) Oceanographic Toolbox is missing. Density calculations will be skipped.'); 
    end

    % --- 1. DATA INGESTION ---
    disp('--- INGESTING MASTER DATA ---');
    T_Master = read_glider_turbo_raw(dir_master);
    if isempty(T_Master)
        error('FAILURE: Master dataset is empty or unreadable.');
    end
    
    disp('--- INGESTING GRAFT 1 ---');
    T_Greffon1 = read_glider_turbo_raw(dir_g1);
    
    disp('--- INGESTING GRAFT 2 (ADCP/UVP6) ---');
    T_Greffon2 = read_glider_turbo_raw(dir_g2);

    % --- 2. DATA FUSION (KNN SPATIAL SYNCHRONIZATION) ---
    disp('Processing data fusion...');
    if ~isempty(T_Greffon1)
        T_Master = greffer_knn_secure(T_Master, T_Greffon1, DIST_MAX_KM, DLIM_Z); 
    end
    if ~isempty(T_Greffon2)
        T_Master = greffer_knn_secure(T_Master, T_Greffon2, DIST_MAX_KM, DLIM_Z); 
    end

    % --- 3. DATA CLEANING & PROCESSING ---
    vars = T_Master.Properties.VariableNames;
    
    % 3.1 Correcting Biological and UVP6 sensors (replace 9999 with NaN)
    idx_bio = find(contains(vars, 'UBAT') | contains(vars, 'POT') | contains(vars, 'UVP6') | contains(vars, 'BIOLUM'));
    for k = 1:length(idx_bio)
        name = vars{idx_bio(k)};
        data = T_Master.(name);
        mask_bad = (abs(data - 9999) < 1e-3);
        if sum(mask_bad) > 0
            T_Master.(name)(mask_bad) = NaN; 
        end
    end

    % 3.2 Dinoflagellates / Zooplankton discrimination (Messié et al. 2019)
    idx_biolum_col = find(contains(vars, 'BIOLUM_POT'), 1);
    if ~isempty(idx_biolum_col)
        nom_col_biolum = vars{idx_biolum_col};
        disp(['Applying Bio-optical filters on: ' nom_col_biolum '...']);
        
        if ~ismember('ProfileNum', T_Master.Properties.VariableNames)
            T_Master.ProfileNum = ones(height(T_Master), 1);
        end
        
        win_bg = 21;      
        win_stats = 107;   
        
        T_Master.Biolum_Dino = NaN(height(T_Master), 1);
        T_Master.Biolum_Zoo  = NaN(height(T_Master), 1);
        
        seuil_bruit = 1.5e7;
        
        liste_profs = unique(T_Master.ProfileNum(~isnan(T_Master.ProfileNum)));
        for p = 1:length(liste_profs)
            idx = (T_Master.ProfileNum == liste_profs(p));
            sig_total = T_Master.(nom_col_biolum)(idx);
            
            if length(sig_total) < win_stats, continue; end
            
            % BASELINE (Dinoflagellates)
            tmp_min = movmin(sig_total, win_bg, 'omitnan');
            min_bg = movmean(tmp_min, 5, 'omitnan'); 
            T_Master.Biolum_Dino(idx) = min_bg;
            med_bg = movmedian(sig_total, win_bg, 'omitnan');
            env_sup = (2 .* med_bg) - min_bg;
            
            % PEAKS (Zooplankton)
            tmp_zoo = zeros(size(sig_total));
            is_flash = sig_total > env_sup;
            tmp_zoo(is_flash) = sig_total(is_flash) - med_bg(is_flash);
            tmp_zoo(tmp_zoo < seuil_bruit) = seuil_bruit; 
            T_Master.Biolum_Zoo(idx) = tmp_zoo;
        end
    end

    % 3.3 Thermodynamic Density (TEOS-10)
    try 
        T_Master = calculer_sigma0_robuste(T_Master); 
    catch
        disp('Skipping density calculation.');
    end

    % --- 4. FINAL SORTING & EXPORT ---
    if ismember('Time', T_Master.Properties.VariableNames)
        T_Master = sortrows(T_Master, 'Time');
    end

    disp('Saving final dataset...');
    save('DATA_FUSION_FINAL_MULTISENSOR.mat', 'T_Master', '-v7.3');
    write_cnv_flexible(T_Master, 'FUSION_MULTISENSOR.cnv');
    disp(['Fusion complete! Processed ' num2str(height(T_Master)) ' rows.']);
end

%% --- LOCAL FUNCTIONS ---

function T_out = read_glider_turbo_raw(folder_path)
    % Parses raw glider telemetry files (.pld1, .raw, .sub)
    file_list = dir(fullfile(folder_path, '**', '*pld1*'));
    file_list = file_list(~[file_list.isdir]);
    
    if isempty(file_list)
        disp('   [EMPTY] No files found.');
        T_out = []; return;
    end
    
    [~, idx] = sort({file_list.name}); file_list = file_list(idx);
    T_out = table();
    disp(['   -> Processing ' num2str(length(file_list)) ' files...']);
    
    for i = 1:length(file_list)
        path = fullfile(file_list(i).folder, file_list(i).name);
        try
            fid = fopen(path, 'r');
            if fid == -1, continue; end
            
            file_content = fread(fid, '*char')';
            fclose(fid);
            
            % Fix broken timestamp linebreaks
            file_content = regexprep(file_content, '(\d{2}/\d{2}/\d{4})[\r\n]+(\d{2}:\d{2}:\d{2})', '$1 $2');
            
            temp_path = tempname;
            fid_temp = fopen(temp_path, 'w');
            fwrite(fid_temp, file_content);
            fclose(fid_temp);
            
            fid = fopen(temp_path, 'r');
            header_str = fgetl(fid);
            if isempty(header_str) || ~ischar(header_str)
                 fclose(fid); delete(temp_path); continue;
            end
            
            if count(header_str, ';') >= count(header_str, ','), delim = ';'; else, delim = ','; end
            header_cols = strsplit(upper(regexprep(header_str, '"', '')), delim);
            header_cols = regexprep(header_cols, '[^A-Z0-9_]', '_'); 
            
            col_map = struct();
            fmt = repmat({'%*s'}, 1, length(header_cols)); 
            
            keys_vital = {'TIME', 'LATITUDE', 'LONGITUDE', 'DEPTH'};
            keys_data  = {'UBAT', 'BIOLUM', 'AROD', 'PAR', 'IRRADIANCE', 'FLBBCD', 'FLNTU', 'CHL', 'SALINITY', 'TEMPERATURE', 'PRESSURE', 'CONDUCTIVITY', 'UVP6', 'AD2CP'};
            
            for c = 1:length(header_cols)
                col_name = header_cols{c};
                if isempty(col_name), continue; end
                
                is_vital = any(contains(col_name, keys_vital));
                is_data  = any(contains(col_name, keys_data));
                
                if is_vital || is_data
                    if contains(col_name, 'TIME') || contains(col_name, 'CLOCK')
                        fmt{c} = '%s'; 
                    else
                        fmt{c} = '%f'; 
                    end
                    col_map.(col_name) = c;
                end
            end
            fmt_str = strjoin(fmt, ' ');
            
            % Fast textscan ingestion
            data_cell = textscan(fid, fmt_str, 'Delimiter', delim, 'TreatAsEmpty', {'NA', 'NaN'}, 'HeaderLines', 0);
            fclose(fid);
            delete(temp_path);
            
            if isempty(data_cell{1}), continue; end
            
            T_temp = table();
            data_idx = 1;
            for c = 1:length(header_cols)
                name = header_cols{c};
                if isfield(col_map, name)
                    raw_col = data_cell{data_idx};
                    if iscell(raw_col)
                         try
                            d_vec = datenum(raw_col, 'dd/mm/yyyy HH:MM:SS.FFF');
                         catch
                            try d_vec = datenum(raw_col); catch, d_vec = NaN(size(raw_col)); end
                         end
                         T_temp.(name) = d_vec;
                    else
                        T_temp.(name) = raw_col;
                    end
                    data_idx = data_idx + 1;
                end
            end
            
            % Metadata Normalization
            my_vars = T_temp.Properties.VariableNames;
            if any(contains(my_vars, 'LATITUDE'))
                idx = find(contains(my_vars, 'LATITUDE'), 1);
                lat_raw = T_temp.(my_vars{idx});
                if nanmean(abs(lat_raw)) > 90
                    d = fix(lat_raw/100); m = abs(lat_raw)-abs(d*100); lat_raw = d + (m/60);
                end
                T_temp.NAV_LATITUDE = lat_raw;
            end
            
            if any(contains(my_vars, 'LONGITUDE'))
                idx = find(contains(my_vars, 'LONGITUDE'), 1);
                lon_raw = T_temp.(my_vars{idx});
                if nanmean(abs(lon_raw)) > 180
                    d = fix(lon_raw/100); m = abs(lon_raw)-abs(d*100); lon_raw = d + (m/60);
                end
                T_temp.NAV_LONGITUDE = lon_raw;
            end
            
            if any(contains(my_vars, 'NAV_DEPTH'))
                T_temp.NAV_DEPTH = T_temp.NAV_DEPTH;
            elseif any(contains(my_vars, 'PRESSURE'))
                 idx = find(contains(my_vars, 'PRESSURE'), 1); 
                 if ~isempty(idx), T_temp.NAV_DEPTH = T_temp.(my_vars{idx}); end
            end
            
            if ~ismember('Time', T_temp.Properties.VariableNames)
                if any(contains(my_vars, 'CLOCK'))
                    idx=find(contains(my_vars, 'CLOCK'),1); T_temp.Time=T_temp.(my_vars{idx});
                elseif any(contains(my_vars, 'TIME'))
                    idx=find(contains(my_vars, 'TIME'),1); T_temp.Time=T_temp.(my_vars{idx});
                end
            end
            
            if ~ismember('NAV_LATITUDE', T_temp.Properties.VariableNames) || ~ismember('Time', T_temp.Properties.VariableNames)
                continue;
            end
            
            T_out = [T_out; T_temp];
        catch ME
            disp(['Error processing ' file_list(i).name ' : ' ME.message]);
        end
    end
    
    if ~isempty(T_out)
        disp(['   -> SUCCESS: ' num2str(height(T_out)) ' records parsed.']);
    end
end

function T_Hote = greffer_knn_secure(T_Hote, T_Greffon, max_dist_km, dZ)
    % Synchronizes data arrays using K-Nearest Neighbors in 3D space
    if isempty(T_Hote) || isempty(T_Greffon), return; end
    req = {'NAV_LATITUDE', 'NAV_LONGITUDE', 'NAV_DEPTH'};
    if ~all(ismember(req, T_Hote.Properties.VariableNames)) || ~all(ismember(req, T_Greffon.Properties.VariableNames)), return; end
    
    LatH = T_Hote.NAV_LATITUDE; LonH = T_Hote.NAV_LONGITUDE; DepthH = T_Hote.NAV_DEPTH;
    LatG = T_Greffon.NAV_LATITUDE; LonG = T_Greffon.NAV_LONGITUDE; DepthG = T_Greffon.NAV_DEPTH;
    
    validG = ~isnan(LatG) & ~isnan(LonG) & ~isnan(DepthG);
    LatG = LatG(validG); LonG = LonG(validG); DepthG = DepthG(validG); T_Greffon = T_Greffon(validG, :);
    
    if isempty(LatG), return; end
    
    try
        CoordsH = [LatH, LonH, DepthH * 0.1]; CoordsG = [LatG, LonG, DepthG * 0.1];
        [idx, ~] = knnsearch(CoordsG, CoordsH);
        
        vars_ignore = {'Time', 'NAV_LONGITUDE', 'NAV_LATITUDE', 'NAV_DEPTH'};
        vars_to_graft = setdiff(T_Greffon.Properties.VariableNames, vars_ignore);
        
        for k = 1:length(vars_to_graft)
            var_name = vars_to_graft{k};
            if ~contains(var_name, {'UBAT','BIO','PAR','FLUO','CHL','SAL','TEMP','O2','DO','CONDUCTIVITY','UVP6','AD2CP'}), continue; end
            
            if ismember(var_name, T_Hote.Properties.VariableNames)
                if sum(~isnan(T_Hote.(var_name)))/height(T_Hote) > 0.05, continue; end
            end
            
            data_src = T_Greffon.(var_name); newData = NaN(height(T_Hote), 1);
            for i = 1:length(idx)
                dist_h_km = distance_haversine(LatH(i), LonH(i), LatG(idx(i)), LonG(idx(i)));
                if dist_h_km <= max_dist_km && abs(DepthH(i) - DepthG(idx(i))) <= dZ
                    newData(i) = data_src(idx(i));
                end
            end
            T_Hote.(var_name) = newData;
        end
    catch
    end
end

function d = distance_haversine(lat1, lon1, lat2, lon2)
    R = 6371; dlat = deg2rad(lat2-lat1); dlon = deg2rad(lon2-lon1);
    a = sin(dlat/2)^2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon/2)^2;
    d = R * (2 * atan2(sqrt(a), sqrt(1-a)));
end

function T = calculer_sigma0_robuste(T)
    T.Sigma0 = NaN(height(T), 1);
    vars = T.Properties.VariableNames;
    idx_S = find(contains(vars, 'SALINITY'), 1); idx_T = find(contains(vars, 'TEMPERATURE'), 1);
    
    if isempty(idx_S) || isempty(idx_T), return; end
    S = T.(vars{idx_S}); Temp = T.(vars{idx_T});
    
    if ismember('NAV_LATITUDE', vars), lat = T.NAV_LATITUDE; else, lat = zeros(height(T),1); end
    if ismember('NAV_LONGITUDE', vars), lon = T.NAV_LONGITUDE; else, lon = zeros(height(T),1); end
    if ismember('NAV_DEPTH', vars), pres = T.NAV_DEPTH; else, pres = zeros(height(T),1); end
    
    mask = ~isnan(S) & ~isnan(Temp) & ~isnan(pres);
    if sum(mask) > 0
        P_safe = pres(mask); P_safe(P_safe < 0) = 0; 
        SA = gsw_SA_from_SP(S(mask), P_safe, lon(mask), lat(mask)); 
        CT = gsw_CT_from_t(SA, Temp(mask), P_safe); 
        T.Sigma0(mask) = gsw_sigma0(SA, CT);
    end
end

function write_cnv_flexible(T, filename)
    bad_str = '-9.99e-29'; 
    vars = T.Properties.VariableNames;
    
    priorities = {'NAV_DEPTH', 'NAV_LONGITUDE', 'NAV_LATITUDE', 'UBAT_BIOLUM_POT', 'Biolum_Dino', 'Biolum_Zoo', 'AROD_FT_DO', 'MPE_PAR_IRRADIANCE', 'FLBBCD_CHL_SCALED', 'Sigma0'};
    adcp_uvp_vars = vars(contains(vars, 'AD2CP') | contains(vars, 'UVP6'));
    adcp_uvp_vars = setdiff(adcp_uvp_vars, priorities);
    final_export_list = [priorities, adcp_uvp_vars];
    
    T_Export = table(); names = {};
    
    if ismember('NAV_DEPTH', vars), T.Depth = T.NAV_DEPTH; end
    if ismember('NAV_DEPTH', final_export_list), final_export_list{strcmp(final_export_list, 'NAV_DEPTH')} = 'Depth'; end 
    
    for i=1:length(final_export_list)
        v = final_export_list{i}; 
        col_name = '';
        
        if ismember(v, T.Properties.VariableNames)
            col_name = v; 
        elseif strcmp(v, 'Depth') && ismember('NAV_DEPTH', T.Properties.VariableNames)
            col_name = 'NAV_DEPTH'; 
        end
        
        if ~isempty(col_name)
            col_data = T.(col_name); 
            col_str = cell(height(T), 1); 
            mask_nan = isnan(col_data); 
            vals_ok = col_data(~mask_nan);
            
            if ~isempty(vals_ok), col_str(~mask_nan) = cellstr(num2str(vals_ok, '%.6f')); end
            if any(mask_nan), col_str(mask_nan) = {bad_str}; end
            
            T_Export.(v) = col_str; names{end+1} = v; 
        end
    end
    
    s_time = datestr(now, 'mmm dd yyyy HH:MM:SS');
    
    if ismember('Time', vars)
        t_start = min(T.Time); 
        dvec_start = datevec(t_start); 
        t0 = datenum(dvec_start(1), 1, 1, 0, 0, 0);
        
        col_data = T.Time - t0; 
        col_str = cellstr(num2str(col_data, '%.6f'));
        
        T_Export.TimeJulian = col_str; 
        names{end+1} = 'TimeJulian'; 
        s_time = datestr(t_start, 'mmm dd yyyy HH:MM:SS');
    end
    
    fid = fopen(filename, 'w');
    fprintf(fid, '* Glider MULTISENSOR FUSION (CTD, UVP6, ADCP, UBAT)\n'); 
    fprintf(fid, '* FileName = %s\n', filename);
    fprintf(fid, '# nquan = %d\n', length(names)); 
    fprintf(fid, '# nvalues = %d\n', height(T));
    fprintf(fid, '# units = metric\n'); 
    fprintf(fid, '# bad_flag = %s\n', bad_str);
    
    for i = 1:length(names), fprintf(fid, '# name %d = %s\n', i-1, names{i}); end
    
    fprintf(fid, '# start_time = %s\n', s_time); 
    fprintf(fid, '*END*\n');
    
    data_cells = table2cell(T_Export)'; 
    n_cols = length(names);
    fmt_str = [repmat('%s;', 1, n_cols-1), '%s\n'];
    fprintf(fid, fmt_str, data_cells{:}); 
    fclose(fid);
end
