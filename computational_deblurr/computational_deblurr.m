%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image deblurring using Wiener Filtering and Richardson-Lucy algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clf; close all;

%% Loading sharp image, adding noise and blurr to it 
% load image
I_orig = im2double(imread('FluorescentCells.jpg'));
imageSize = [size(I_orig,1) size(I_orig,2)];

% load PSF
psf = fspecial('gaussian', [15 15], 1.5);

% compute OTF
otf = psf2otf(psf, imageSize);

% number of photons
nphotons = 100;
I = I_orig.*nphotons;

% define function handle for 2D convolution
Afun    = @(x) ifft2( fft2(x).*otf, 'symmetric');
Atfun   = @(x) ifft2( fft2(x).*conj(otf), 'symmetric');

% simulate blurry measurements
for c=1:3
    b(:,:,c) = Afun(I(:,:,c));
end
% make sure there are absolutely no negative values in here
b(b<0)=0;
% now simulate poisson noise
b = poissrnd(b);


%% Wiener filtering deblurring
close all;

% plot noisy image
subplot(1,2,1);
imshow(b./nphotons);
title('Noisy image');
set(gca,'fontsize', 14);
% invserse filter image here
fft_b = fft2(b);
snr = abs(fft2(sqrt(b))); %mean value is pixel value, stdev is sqrt(pixel value). So snr = pixval/sqrt(pixval) = sqrt(pixval)
% snr = sqrt(b);
recons = zeros([imageSize 3]);

for chan = 1:3 
    damping = (abs(otf).^2)./((abs(otf).^2)+(1./snr(:,:,chan)));
    recons(:,:,chan) = ifft2(damping.*fft_b(:,:,chan)./otf);
end

recons_r = recons(:,:,1);
% recons = ifft2(damping.*fft2(b)./otf);
% getting peak_snr
mse = mean(real(recons(:)./nphotons-I_orig(:)).^2);
peak_snr = 10*log10(1/mse)
% plot reconstructed image
subplot(1,2,2);
imshow(real(recons./nphotons));
title({'Reconstructed image'; 'with Wiener deblurring'});
set(gca,'fontsize', 14);


%% Richardson-Lucy algorithm
close all;

n_iters = 25;
recons_lucy = 1e-3.*ones([imageSize 3]);
% imshow(recons_lucy)
mse_lucy = zeros(n_iters);
log_like = zeros(n_iters);

for it = 1:n_iters
    for chan = 1:3
        factor = Atfun(b(:,:,chan)./Afun(recons_lucy(:,:,chan))) ./Atfun(ones(imageSize));
        recons_lucy(:,:,chan) = factor.*recons_lucy(:,:,chan);
        factor = zeros(size(factor));
    end
%     imshow(recons_lucy)
%     input('enter')
    recons_lucy(isnan(recons_lucy)) = 1e-3;
    recons_lucy(recons_lucy<=0) = 1e-3;
    for chan = 1:3
        log_like(it) = log_like(it) + sum(sum(log(Afun(recons_lucy(:,:,chan))).*b(:,:,chan))) + ...
            - sum(sum(Afun(recons_lucy(:,:,chan)))) ...
            - sum(sum(log(factorial(b(:,:,chan)))));
    end
    mse_lucy(it) = mean(real(recons_lucy(:)./nphotons-I_orig(:)).^2);
    disp(['iters remaining: ' int2str(n_iters-it)])
end

peak_snr_lucy = 10*log10(1/mse_lucy(it))
subplot(2,2,1);
imshow(real(b./nphotons));
title('Noisy image');
set(gca,'fontsize', 14);

subplot(2,2,2);
imshow(real(recons_lucy./nphotons));
title('Reconstructed image');
set(gca,'fontsize', 14);

subplot(2,2,3);
plot(mse_lucy);
title('MSE vs iteration num');
set(gca,'fontsize', 14);

subplot(2,2,4);
plot(log_like);
title('Log likelihood vs iteration num');
set(gca,'fontsize', 14);



%% Richardson-Lucy with TV prior
close all;

n_iters = 25;
recons_lucy_tv = 1e-3.*ones([imageSize 3]);
% imshow(recons_lucy_tv)
mse_lucy = zeros(n_iters);
log_like = zeros(n_iters);
grad_order = 1;
lam = 0.04
for it = 1:n_iters
%     it
    for chan = 1:3
        grad_x = diff(recons_lucy_tv(:,:,chan), grad_order, 2); grad_x(:,end+1,:) = 0;
        grad_y = diff(recons_lucy_tv(:,:,chan), grad_order, 1); grad_y(end+1,:,:) = 0;
        grad = sqrt(grad_x.^2 + grad_y.^2);
        tv_term = (grad_x./abs(grad_x) + grad_y./abs(grad_y));
        tv_term(isnan(tv_term)) = 0;
        factor = Atfun(b(:,:,chan)./Afun(recons_lucy_tv(:,:,chan))) ./ ...
            ( Atfun(ones(imageSize)) - lam.*tv_term );
        recons_lucy_tv(:,:,chan) = factor.*recons_lucy_tv(:,:,chan);
        factor = zeros(size(factor));
    end
%     imshow(recons_lucy_tv)
%     input('enter')
    recons_lucy_tv(isnan(recons_lucy_tv)) = 1e-3;
    recons_lucy_tv(recons_lucy_tv<=0) = 1e-3;
    for chan = 1:3
        log_like(it) = log_like(it) + sum(sum(log(Afun(recons_lucy_tv(:,:,chan))).*b(:,:,chan))) + ...
            - sum(sum(Afun(recons_lucy_tv(:,:,chan)))) ...
            - sum(sum(log(factorial(b(:,:,chan))))) ...
            - lam*sum(sum(abs(grad)));
    end
    
    mse_lucy(it) = mean(real(recons_lucy_tv(:)./nphotons-I_orig(:)).^2);
    disp(['iters remaining: ' int2str(n_iters-it)])
end

peak_snr_lucy = 10*log10(1/mse_lucy(it))
subplot(2,2,1);
imshow(real(b./nphotons));
title('Noisy image');
set(gca,'fontsize', 14);

subplot(2,2,2);
imshow(real(recons_lucy_tv./nphotons));
title('Reconstructed image');
set(gca,'fontsize', 14);

subplot(2,2,3);
plot(mse_lucy);
title('MSE vs iteration num');
set(gca,'fontsize', 14);

subplot(2,2,4);
plot(log_like);
title('Log likelihood vs iteration num');
set(gca,'fontsize', 14);
