% To maintain code simplicity and readability, the current implementation utilizes several OpenCV integrated functions. 
% It is primarily aimed at demonstrating algorithm performance and does not necessarily reflect its actual computational efficiency.

% Input:
% img_gray      - Grayscale image
% kl            - OpenCV formatted line segments
% radius        - Radius for line segment ROS
% g             - Spatial granularity

% Output:
% kl            - OpenCV formatted line segments
% descs         - Line segment descriptors

function [kl, descs] = iilb(img_gray, kl, radius, g)

sobel_x             = [-1  0  1;  -2 0 2; -1 0 1];
sobel_y             = [-1 -2 -1;   0 0 0;  1 2 1];

[base_rois, c_idx]  = get_rois_lltb(zeros(2*radius+1, 100), g);

desc_dim            = size(c_idx, 1)*4;
template            = repmat('0', desc_dim, 1);
descs               = uint8(zeros(size(kl, 1), ceil(desc_dim/8)));

if ~isempty(kl)
    kl_lengths          = vecnorm(kl(:, 3:4)-kl(:, 1:2), 2, 2);
    kl_angles           = atan((kl(:, 4)-kl(:, 2)) ./ (kl(:, 3)-kl(:, 1)));
end

for i = 1:size(kl, 1)
    
    rect             = [];
    rect.center      = (kl(i, 1:2) + kl(i, 3:4)) / 2;
    rect.size        = [kl_lengths(i)+2, 2*radius+1+2];  
    rect.angle       = kl_angles(i)*180/pi;
    
    scr_pts          = boxPoints(rect);
    
    h                = rect.size(2);
    w                = rect.size(1);
    dst_pts          = round([0, h-1; 0 0; w-1, 0; w-1, h-1]);
    
    M                = getPerspectiveTransform(scr_pts, dst_pts);
    gray_rect        = warpPerspective(img_gray, M, 'DSize', [w, h], 'BorderType', 'Reflect');
    
    gx                   = filter2D(single(gray_rect), sobel_x);
    gy                   = filter2D(single(gray_rect), sobel_y);
    gx_img               = uint16(abs(gx));
    gx_rect              = gx_img(2:end-1, 2:end-1);
    gy_img               = uint16(abs(gy));
    gy_rect              = gy_img(2:end-1, 2:end-1);
    g_ori_img            = uint16((atan2d(-gy, gx)+180));
    g_ori_rect           = g_ori_img(2:end-1, 2:end-1);
    
    gray_rect            = gray_rect(2:end-1, 2:end-1);
    
    rois                    = base_rois;
    rois(:, 3)              = size(gray_rect, 2);
    
    [row, col]              = size(gray_rect);
    nums                    = (rois(:, 3)-rois(:, 1)+1).*(rois(:, 4)-rois(:, 2)+1);
    inds_2_2                = sub2ind([row+1, col+1], rois(:, 4)+1,       rois(:, 3)+1);
    inds_1_1                = sub2ind([row+1, col+1], rois(:, 2),         rois(:, 1));
    inds_1_2                = sub2ind([row+1, col+1], rois(:, 4)+1,       rois(:, 1));
    inds_2_1                = sub2ind([row+1, col+1], rois(:, 2),         rois(:, 3)+1);
    linear_indices          = [inds_1_1 inds_1_2 inds_2_1 inds_2_2];
    
    descs(i, :)             = my_desc_lltb(gray_rect, gx_rect, gy_rect, g_ori_rect, linear_indices, c_idx, nums, template);
    
end

end

%%
function [rois, c_idx] = get_rois_lltb(rect, g)

[row, col] = size(rect);

rois       = cell(1, g);
intra_idx  = cell(1, g);
inter_idx  = cell(1, g-1);

acc_num    = cumsum(2.^(0:g-1))-1;

for i = 1:g
    step            = (row-1) / (2^i);
    r_list          = 1:step:row;
    temp_rois       = zeros(length(r_list)-1, 4);    
    temp_intra_idx  = zeros(length(r_list)-2, 2);
    
    for j = 1:length(r_list)-1
        temp_rois(j, :)   = [1 r_list(j) col r_list(j+1)];
    end
    rois{i}         = temp_rois;
    
    % adjacent 
    for j = 1:length(r_list)-2
        temp_intra_idx(j, :) = [j, j+1];
    end
    
    temp_intra_idx  = temp_intra_idx + acc_num(i);
    intra_idx{i}    = temp_intra_idx;
    
    % symmetric
    if i > 1
        temp_inter_idx  = zeros(((length(r_list)-1)/2-1), 2);
        for j = 1:((length(r_list)-1)/2-1)
            temp_inter_idx(j, :) = [j, length(r_list)-j];
        end
        temp_inter_idx  = temp_inter_idx + acc_num(i);
        inter_idx{i-1}    = temp_inter_idx;
    end
end

rois        = round(cell2mat(rois'));
c_idx       = cell2mat(intra_idx');

end

%%
function [descs] = my_desc_lltb(gray_rect, gx_rect, gy_rect, ori, linear_indices, c_idx, nums, template)

[desc_gray]     = desc_unit(gray_rect, linear_indices, c_idx, nums);
[desc_gx]       = desc_unit(gx_rect, linear_indices, c_idx, nums);
[desc_gy]       = desc_unit(gy_rect, linear_indices, c_idx, nums);
[desc_ori]      = desc_unit(ori, linear_indices, c_idx, nums);

inliers         = [desc_gray; desc_gx; desc_gy; desc_ori];

padding         = mod(length(inliers), 8);
if padding > 0
    inliers         = [inliers; true(8-padding, 1)];
end

template(inliers)   = '1';

template            = reshape(template, 8, [])';
descs               = uint8(bin2dec(template))';

end

%%
function [desc] = desc_unit(rect, linear_indices, c_idx, nums)

i_rect          = integral(rect);
temp_values_raw = i_rect(linear_indices(:, 4)) + i_rect(linear_indices(:, 1)) - i_rect(linear_indices(:, 2)) - i_rect(linear_indices(:, 3));
temp_values_raw = double(temp_values_raw) ./ nums;
desc            = temp_values_raw(c_idx(:, 1)) > temp_values_raw(c_idx(:, 2));

end