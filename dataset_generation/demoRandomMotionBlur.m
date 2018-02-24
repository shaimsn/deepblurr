%
%
% this script illustrates the generation of random motion PSFs and of motion blurred pictures by setting the exposure
% time. The image formation model and the description of PSF generation is reported in [Boracchi Foi and 2011] and [Boracchi and Foi 2012]
%
%
%
% References
% [Boracchi and Foi 2012] Giacomo Boracchi and Alessandro Foi, "Modeling the Performance of Image Restoration from Motion Blur"
%  Image Processing, IEEE Transactions on. vol.21, no.8, pp. 3502 - 3517, Aug. 2012, doi:10.1109/TIP.2012.2192126
% Preprint Available at http://home.dei.polimi.it/boracchi/publications.html
%
% [Boracchi and Foi 2011] Giacomo Boracchi and Alessandro Foi, "Uniform motion blur in Poissonian noise: blur/noise trade-off"
%  Image Processing, IEEE Transactions on. vol. 20, no. 2, pp. 592-598, Feb. 2011 doi: 10.1109/TIP.2010.2062196
% Preprint Available at http://home.dei.polimi.it/boracchi/publications.html
%
% December 2012
%
% Giacomo Boracchi*, Alessandro Foi**
% giacomo.boracchi@polimi.it
% alessandro.foi@tut.fi
% * Politecnico di Milano
% **Tampere University of Technology

close all
clear
clc
dir_name = '../crop_drinks_dataset/';
drinks_names = dir(([dir_name '*.JPEG']));
n_im = length(drinks_names);
%%
for i = 1:n_im
    
    do_show = 1;
    
    % trajectory curve parameters
    PSFsize = 17;
    anxiety = 0.005;
    numT = 2000;
    MaxTotalLength = 64;
    
    % PSF parameters
    T = [0.0625 , 0.25 , 0.5, 1]; % exposure Times
    do_centerAndScale = 0;
    
    % noise paramters
    lambda = 2048;
    sigmaGauss = 0.05;
    
    % load original image
    im_num = drinks_names(i).name(1:end-5);
    y = im2double(imread([dir_name im_num '.JPEG']));
    
    %% Generate Random Motion Trajectory
    TrajCurve = createTrajectory(PSFsize, anxiety, numT, MaxTotalLength, do_show);
    
    %% Sample TrajCurve and Generate Motion Blur PSF
    PSFs = createPSFs(TrajCurve, PSFsize,  T , do_show , do_centerAndScale);
    
    %% Generate the sequence of motion blurred observations
    zeroCol = [];%zeros(size(y,1) , 5);
    paddedImage = zeroCol;
    for ii = 1 : numel(PSFs)
        z{ii} = createBlurredRaw(y, PSFs{ii}, lambda, sigmaGauss);
%         figure();
%         imshow(z{ii}./max(z{ii}(:)), []),title(['image having exposure time ', num2str(T(ii))]);
        imTemp = z{ii}./max(z{ii}(:));
        imwrite(imTemp, ['../blur_drinks_dataset/' im_num '_b' int2str(ii) '.JPEG'])
        imwrite(PSFs{ii}, ['../kernels_dataset/' im_num '_k' int2str(ii) '.JPEG'])
%         imTemp(1 : size(PSFs{ii}, 1), 1 : size(PSFs{ii} , 2)) = PSFs{ii}./max(PSFs{ii}(:));
%         paddedImage=[paddedImage, imTemp, zeroCol];
    end
    i
end
% figure(), imshow(paddedImage,[]),title('Sequence of observations, PSFs is shown in the upper left corner');
% imwrite(paddedImage, 'pdimage.png', 'png');


% folder:
% crops_drinks_dataset/ -> cropped images. naming: c_im#
% blur_drinks_dataset/ -> 4 blurry images per image. naming: im98_b1, im98_b2, im99_b1, im_99_b2
% kernels_dataset -> 4 kernels per image. naming: im98_k1, im98_k2, im99_k1, im_99_k2



% c = cropped