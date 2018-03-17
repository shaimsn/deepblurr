function recons = wienerDeconv(b, kern_psf, snr)
% b = noisy input image
% lambda = 1/SNR
% plot noisy image
% subplot(1,2,1);
% imshow(b);
% title('Noisy image');
% set(gca,'fontsize', 14);

im_size = size(b);
kern_otf = psf2otf(kern_psf,im_size(1:2));
fft_b = fft2(b);
recons = zeros(size(b));


for chan = 1:3 
    damping = (abs(kern_otf).^2)./((abs(kern_otf).^2)+(1./snr));
    recons(:,:,chan) = real(ifft2(damping.*fft_b(:,:,chan)./kern_otf));
end

recons_r = recons(:,:,1);


% plot reconstructed image
% subplot(1,2,2);
% imshow(real(recons));
% title({'Reconstructed image'; 'with Wiener deblurring'});
% set(gca,'fontsize', 14);

end