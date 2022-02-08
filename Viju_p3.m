
%% color image denoising by Fast high dimensional nlmeans filter and method noise thresholding with wavelet transform
%Fast High-Dimensional Bilateral and
%Nonlocal Means Filtering
%Pravin Nair, Student Member, IEEE, and Kunal N. Chaudhury, Senior Member, IEEE


%function results=nlmwavelet(image)
%Iact=im2double(image);
clear
clc

 [filename, user_canceled] = imgetfile;
 Iact  =  im2double(imread(filename));
[m,n,d]=size(Iact);
sigma=0.1;
S=40;K=3;
fast_flag=1;
% [filename, user_canceled] = imgetfile;
%  Img_noisy  =  im2double(imread(filename));
  Img_noisy=Iact+sigma*randn(m,n,d);
I2=Iact./256;

%%


%% wavelet denoising only
%wimg = wdenoise2(Img_noisy,'Wavelet','db8','DenoisingMethod','SURE','Colorspace','Original');  %'haar', 'dbN', 'fkN', 'coifN', or 'symN'
wimg = wdenoise2(Img_noisy,'Wavelet','sym4','Colorspace','Original');  %'haar', 'dbN', 'fkN', 'coifN', or 'symN'

%% Kmeans filtering

% Done in two steps : Clustering and Filtering
tic,
Cluster=31;
pcadim=25;

% Clustering
Apca=compute_pca(I2, K, pcadim);
pcadim=size(Apca,3);
Apcares=imresize(Apca,[256 256]);
Ares=reshape(Apcares,size(Apcares,1)*size(Apcares,2),pcadim);
Centre=kmeans_recursive(Ares,Cluster);

% Filtering
spatialtype='box';
convmethod='O1'; % 'matlab' for matlab convolutions and 'O1' for O(1) convolutions
%convmethod='matlab';
Ikmean=fastKmeansfiltapproxinter(I2,S,3.5*sigma/256,Centre,spatialtype,convmethod,fast_flag,Apca);      % nlm kmeans
Ikmean=Ikmean.*255;
Ikmean(Ikmean>=255)=255;
Ikmean(Ikmean<=0)=0;
toc
Tkmeans=toc;
fprintf('non local means by Kmeans complete with %d clusters \n',Cluster);
fprintf('time for non local means (ms)=%3.0f \n',Tkmeans*1000);

%% Displaying noisy and filtered image
% figure(1);
% subplot(141) ,imshow(Iact),title('Original Image')
% subplot(142),imshow(Img_noisy),title('Noisy image')
% subplot(143);imshow(Ikmean);title(['fast nlm with ',num2str(Cluster),' clusters']);

img_method=Iact-Ikmean;
figure;imshow(img_method);
tic
 fimg = wdenoise2(img_method,2,'Wavelet','sym4','DenoisingMethod','SURE','Colorspace','Original');% ,%  %'haar', 'dbN', 'fkN', 'coifN', or 'symN'
 %IMDEN = %WDENOISE2(...,'DenoisingMethod',DMETHOD); %'Bayes','FDR','Minimax','SURE', or
%   'UniversalThreshold'.default Bayes;
%fimg=wdenoise2(img_method,'Wavelet','db8','DenoisingMethod','SURE','NoiseEstimate','LevelIndependent','Colorspace','PCA');
toc

% IMDEN = %WDENOISE2(...,'ThresholdRule',THRESHRULE); %For 'SURE','Minimax', and 'UniversalThreshold', valid options are
%   'Soft' or 'Hard'. The default is 'Soft'.
%IMDEN = WDENOISE2(...,'ColorSpace',CSPACE); 'Original', or 'PCA'
Img_output=Ikmean+fimg;
%Img_output=imguidedfilter(Img_output);
% figure(2),
%subplot(131) ,imshow(Iact),title('Original Image')
%subplot(132),imshow(Img_noisy),title('Noisy image')
figure(1);
set(gcf,'color','w');
set(gca,'FontSize',10)
%subplot(131) ,imshow(Iact),xlabel('Test Image ')
%montage(Img_noisy,Ikmean,wimg,Img_output,  'BorderSize', 6);
 subplot(121),imshow(Img_noisy),  xlabel('Noisy (10%)'),set(gca,'xtick',[],'ytick',[])  ;     %title('Noisy(20%)');
 %subplot(222);imshow(Ikmean);xlabel('Fast HDnlm'),set(gca,'xtick',[],'ytick',[]);
 %subplot(223);imshow(wimg);xlabel('WDenoise'),set(gca,'xtick',[],'ytick',[]);
 subplot(122) ,imshow(Img_output),xlabel('Denoised'),set(gca,'xtick',[],'ytick',[]);
 figure,imshow(Img_output),xlabel('Denoised'),
%  ha=get(gcf,'children');
%  set(ha(1),'position',[.5 .1 .4 .4])
%  set(ha(2),'position',[.1 .1 .4 .4])
%  set(ha(3),'position',[.5 .5 .4 .4])
%  set(ha(4),'position',[.1 .5 .4 .4])

%evaluate_denoising_metrics(Iact,Img_noisy,Img_output);
%% 
fastPSNR = psnr(Ikmean,Iact);
 disp(['FastHD PSNR = ',num2str(fastPSNR)])
 WDPSNR = psnr(wimg,Iact);
 disp(['WDPSNR = ',num2str(WDPSNR)])
 ProPSNR = psnr(Img_output,Iact);
 disp(['Pro PSNR = ',num2str(ProPSNR)])
 
fastSSIM = ssim(Ikmean,Iact);
 disp(['FastHD SSIM = ',num2str(fastSSIM)])
 

WDSSIM = ssim(wimg,Iact);
 disp(['WDSSIM = ',num2str(WDSSIM)])

  
ProSSIM = ssim(Img_output,Iact);
 disp(['Pro SSIM = ',num2str(ProSSIM)])
  
% Ea1 = imbinarize(Iact);
% Ea = imcomplement(Ea1);
% Ed1 = imbinarize(Img_output);
% Ed = imcomplement(Ed1);
% F = pratt(Ea,Ed);
%  disp(['pratts figure of merit is = ',num2str(F)]) 
  
 
  
