%%% Generated Dataset Structure
% crops_drinks_dataset/ -> cropped images. naming: c_im#
% blur_drinks_dataset/ -> 4 blurry images per image. naming: im98_b1, im98_b2, im99_b1, im_99_b2
% kernels_dataset -> 4 kernels per image. naming: im98_k1, im98_k2, im99_k1, im_99_k2
% kernels_normalized_dataset -> normalized by highest value. For viewing purposes
% stack_drinks_dataset/ -> stacks of 15 Wiener deconvolved versions of a blurry image


close all
clear
clc

dir_name = 'crop_drinks_dataset/';
drinks_names = dir(([dir_name '*.JPEG']));
n_im = length(drinks_names);

%%% Range of SNR values to be used in Wiener Filtering. 15 values
snrs = [9:4:68];

for i = 1:n_im
    %%% Defining Parameters and Loading Sharp Image  
    do_show = 1;
    
    % trajectory curve parameters
    PSFsize = 34;
    anxiety = 0.005;
    numT = 2000;
    MaxTotalLength = 64;
    
    % PSF parameters
    T = [0.125 , 0.25 , 0.5, 1]; % exposure Times
    do_centerAndScale = 0;
    
    % noise paramters
    lambda = 2048;
    sigmaGauss = 0.05;
    
    % load sharp image
    im_num = drinks_names(i).name(1:end-5);
    y = im2double(imread([dir_name im_num '.JPEG']));
    
    %%% Generating Random Motion Trajectory
    TrajCurve = createTrajectory(PSFsize, anxiety, numT, MaxTotalLength, do_show);
    
    %%% Sample TrajCurve to Generate Four Motion Blur PSFs. Saved on PSFs
    PSFs = createPSFs(TrajCurve, PSFsize,  T , do_show , do_centerAndScale);
    
    %%% Populate Dataset Folders
    zeroCol = [];
    paddedImage = zeroCol;
    %   plotting clean image: subplot(2,2,4); imshow(y); title('Clean image'); set(gca,'fontsize', 14);
    for ii = 1 : numel(PSFs)
        %%% Create blurred image by convolving PSF to image
        z{ii} = createBlurredRaw(y, PSFs{ii}, lambda, sigmaGauss);
        imTemp = z{ii}./max(z{ii}(:));
        imwrite(imTemp, ['blur_drinks_dataset/' im_num '_b' int2str(ii) '.png'])
% %     plotting kernel: subplot(2,2,1); imshow(PSFs{ii}/max(max(max(PSFs{ii}))));
%       plotting blurred image: subplot(2,2,2); imshow(imTemp); title('Noisy image'); set(gca,'fontsize', 14);
        
        %%% Creating Wiener Stack of the image
        stack_size = size(imTemp);
        stack_size(2) = stack_size(2)*15;
        wienerStack = zeros(stack_size);
        for ix = 0:length(snrs)-1
            wienerStack(:,(ix*256+1):(ix*256)+256,:) = wienerDeconv(imTemp, PSFs{ii}, snrs(ix+1)); 
%           plotting wiener stack: subplot(2,2,3); imshow(wienerStack(:,(ix*256+1):(ix*256)+256,:)); title({'Reconstructed image'; 'with Wiener deblurring'}); set(gca,'fontsize', 14);
        end
        
        %%% Saving images into respective folders
        imwrite(wienerStack, ['stack_drinks_dataset/' im_num '_ws' int2str(ii) '.png'])
        imwrite(PSFs{ii}, ['kernels_dataset/' im_num '_k' int2str(ii) '.png'])
        imwrite(PSFs{ii}/max(max(max(PSFs{ii}))), ['kernels_normalized_dataset/' im_num '_k' int2str(ii) '.png'])
    end
    i
end
