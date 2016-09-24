function crop_images(src_folder, dst_folder, truncation_distr_file, single_thread)

if nargin < 4
    single_thread = 0;
end
if single_thread
    num_workers = 0;
else
    num_workers = 24;
end

image_files = rdir(fullfile(src_folder,'*/*.png'));
keypoint2d_files = rdir(fullfile(src_folder,'*/*.csv'));
image_num = length(image_files);
fprintf('%d images in total.\n', image_num);
if image_num == 0
    return;
end
rng('shuffle');
truncationParameters = importdata(truncation_distr_file);
truncationParametersSub = truncationParameters(randi([1,length(truncationParameters)],1,image_num),:);

fprintf('Start cropping at time %s...it takes for a while!!\n', datestr(now, 'HH:MM:SS'));
report_num = 80;
fprintf([repmat('.',1,report_num) '\n\n']);
report_step = floor((image_num+report_num-1)/report_num);
t_begin = clock;
%for i = 1:image_num
parfor(i = 1:image_num, num_workers)
    src_image_file = image_files(i).name;
    keypoint2d_file = keypoint2d_files(i).name;
    try
        [I, ~, alpha] = imread(src_image_file);       
    catch
        fprintf('Failed to read %s\n', src_image_file);
    end
    
    fid = fopen(keypoint2d_file, 'r');
    data = textscan(fid, '%s %d %d', 'Delimiter', ',');
    fclose(fid);
    keypoint_name = data{1};
    keypoint_x = data{2};
    keypoint_y = data{3};

    [alpha, top, bottom, left, right] = crop_gray(alpha, 0, truncationParametersSub(i,:));
    I = I(top:bottom, left:right, :);
    keypoint_name_cr = {};
    keypoint_x_cr = [];
    keypoint_y_cr = [];
    for j=1:numel(keypoint_name)
        if keypoint_y(j)+1 >= top && keypoint_y(j)+1 <= bottom  && keypoint_x(j)+1 >= left && keypoint_x(j)+1 <= right
            keypoint_y_new = keypoint_y(j)-top+1;
            keypoint_x_new = keypoint_x(j)-left+1;
            keypoint_name_cr = {keypoint_name_cr{:} keypoint_name{j}};
            keypoint_x_cr = [keypoint_x_cr; keypoint_x_new];
            keypoint_y_cr = [keypoint_y_cr; keypoint_y_new];
        end
    end

    if numel(I) == 0
        fprintf('Failed to crop %s (empty image after crop)\n', src_image_file);
    elseif numel(keypoint_name_cr) == 0
        fprintf('Failed to find keypoints in crop %s\n', src_image_file);
    else
        dst_image_file = strrep(src_image_file, src_folder, dst_folder);
        [dst_image_file_folder, ~, ~] = fileparts(dst_image_file);
        if ~exist(dst_image_file_folder, 'dir')
            mkdir(dst_image_file_folder);
        end
        imwrite(I, dst_image_file, 'png', 'Alpha', alpha);
        
        dst_keypoint_file = strrep(dst_image_file, '.png', '_keypoint2d.csv');
        fid = fopen(dst_keypoint_file, 'w');
        for j=1:numel(keypoint_name_cr)
            fprintf(fid, '%s,%d,%d\n', keypoint_name_cr{j}, keypoint_x_cr(j), keypoint_y_cr(j));
        end
        fclose(fid);
    end
    
    if mod(i, report_step) == 0
        fprintf('\b|\n');
    end
end      
t_end = clock;
fprintf('%f seconds spent on cropping!\n', etime(t_end, t_begin));
end
