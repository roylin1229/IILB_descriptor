clc;clearvars;close all;addpath(".\mexopencv\");

image1                                  = imread('1.ppm');
image2                                  = imread('6.ppm');

detector                                = BinaryDescriptor();

[kl_lsd1]                               = detector.detect(rgb2gray(image1));
[kl_lsd1]                               = ref_line(kl_lsd1);
% To simplify, identical line segments are used in both images (with differences from the literature).
[kl_lsd2]                               = kl_lsd1;

% lsd + lbd (float)
[desc_lbd_float1, kl_lbd_float1]        = detector.compute(rgb2gray(image1), kl_lsd1, 'ReturnFloatDescr', true);
[desc_lbd_float2, kl_lbd_float2]        = detector.compute(rgb2gray(image2), kl_lsd2, 'ReturnFloatDescr', true);
kl_lbd_float1                           = [vertcat(kl_lbd_float1.startPoint) vertcat(kl_lbd_float1.endPoint)] + 1;
kl_lbd_float2                           = [vertcat(kl_lbd_float2.startPoint) vertcat(kl_lbd_float2.endPoint)] + 1;

% lsd + lbd (binary)
[desc_lbd_bin1, kl_lbd_bin1]            = detector.compute(rgb2gray(image1), kl_lsd1, 'ReturnFloatDescr', false);
[desc_lbd_bin2, kl_lbd_bin2]            = detector.compute(rgb2gray(image2), kl_lsd2, 'ReturnFloatDescr', false);
kl_lbd_bin1                             = [vertcat(kl_lbd_bin1.startPoint) vertcat(kl_lbd_bin1.endPoint)] + 1;
kl_lbd_bin2                             = [vertcat(kl_lbd_bin2.startPoint) vertcat(kl_lbd_bin2.endPoint)] + 1;

% lsd + iilb (binary)
kl_iilb1                                = [vertcat(kl_lsd1.startPoint) vertcat(kl_lsd1.endPoint)] + 1;
kl_iilb2                                = [vertcat(kl_lsd2.startPoint) vertcat(kl_lsd2.endPoint)] + 1;
[kl_iilb1, descs_iilb1]                 = iilb(image1, kl_iilb1, 64, 5);
[kl_iilb2, descs_iilb2]                 = iilb(image2, kl_iilb2, 64, 5);

% match descriptors
[idx_lbd_float]                         = match_line(false, desc_lbd_float1, desc_lbd_float2, kl_lbd_float1, kl_lbd_float2);
[idx_lbd_binary]                        = match_line(true, desc_lbd_bin1, desc_lbd_bin2, kl_lbd_bin1, kl_lbd_bin2);
[idx_iilb]                              = match_line(true, descs_iilb1, descs_iilb2, kl_iilb1, kl_iilb2);

%% display results
figure
set(gcf, 'Position', get(0, 'Screensize'));

subplot(2,3,1)
imshow(image1)
hold on
line([kl_lbd_float1(idx_lbd_float(:, 1), 1) kl_lbd_float1(idx_lbd_float(:, 1), 3)]', ...
     [kl_lbd_float1(idx_lbd_float(:, 1), 2) kl_lbd_float1(idx_lbd_float(:, 1), 4)]', 'Color', 'r', 'LineWidth', 1);
legend(['matched lines: ', num2str(size(idx_lbd_float, 1))], 'FontSize', 12);

subplot(2,3,4)
imshow(image2)
hold on
line([kl_lbd_float2(idx_lbd_float(:, 2), 1) kl_lbd_float2(idx_lbd_float(:, 2), 3)]', ...
     [kl_lbd_float2(idx_lbd_float(:, 2), 2) kl_lbd_float2(idx_lbd_float(:, 2), 4)]', 'Color', 'r', 'LineWidth', 1);
legend(['matched lines: ', num2str(size(idx_lbd_float, 1))], 'FontSize', 12);

subplot(2,3,2)
imshow(image1)
hold on
line([kl_lbd_bin1(idx_lbd_binary(:, 1), 1) kl_lbd_bin1(idx_lbd_binary(:, 1), 3)]', ...
     [kl_lbd_bin1(idx_lbd_binary(:, 1), 2) kl_lbd_bin1(idx_lbd_binary(:, 1), 4)]', 'Color', 'r', 'LineWidth', 1);
legend(['matched lines: ', num2str(size(idx_lbd_binary, 1))], 'FontSize', 12);

subplot(2,3,5)
imshow(image2)
hold on
line([kl_lbd_bin2(idx_lbd_binary(:, 2), 1) kl_lbd_bin2(idx_lbd_binary(:, 2), 3)]', ...
     [kl_lbd_bin2(idx_lbd_binary(:, 2), 2) kl_lbd_bin2(idx_lbd_binary(:, 2), 4)]', 'Color', 'r', 'LineWidth', 1);
legend(['matched lines: ', num2str(size(idx_lbd_binary, 1))], 'FontSize', 12);

subplot(2,3,3)
imshow(image1)
hold on
line([kl_iilb1(idx_iilb(:, 1), 1) kl_iilb1(idx_iilb(:, 1), 3)]', ...
     [kl_iilb1(idx_iilb(:, 1), 2) kl_iilb1(idx_iilb(:, 1), 4)]', 'Color', 'r', 'LineWidth', 1);
legend(['matched lines: ', num2str(size(idx_iilb, 1))], 'FontSize', 12);

subplot(2,3,6)
imshow(image2)
hold on
line([kl_iilb2(idx_iilb(:, 2), 1) kl_iilb2(idx_iilb(:, 2), 3)]', ...
     [kl_iilb2(idx_iilb(:, 2), 2) kl_iilb2(idx_iilb(:, 2), 4)]', 'Color', 'r', 'LineWidth', 1);
legend(['matched lines: ', num2str(size(idx_iilb, 1))], 'FontSize', 12);



%%
function [index_pairs] = match_line(is_bin_desc, ref_desc, temp_desc, ref_loc, temp_loc)

if is_bin_desc
    matcher             = DescriptorMatcher('BFMatcher', 'NormType', 'Hamming', 'CrossCheck', true);
else
    matcher             = DescriptorMatcher('BFMatcher', 'NormType', 'L2', 'CrossCheck', true);
end

matches                 = matcher.match(ref_desc, temp_desc);
index_pairs             = [[matches.queryIdx]', [matches.trainIdx]'];
index_pairs             = index_pairs + 1;

diff                    = (ref_loc(index_pairs(:, 1), 1:4) - temp_loc(index_pairs(:, 2), 1:4));
valid_idx               = sqrt(sum(diff.^2, 2)) < 1;
index_pairs             = index_pairs(valid_idx, :);

end


%%
function [kl_float] = ref_line(kl_float)

line_length                 = [kl_float.lineLength];
inliers                     = line_length > 20;
kl_float                    = kl_float(inliers);

for i = 1:length(kl_float)
    kl_float(i).class_id = i-1;
end

end