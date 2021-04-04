imgPath='E:/fmh/Low_light/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train_rsz/train_H/';
mapPath='E:/fmh/Low_light/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train_rsz/train_M/';
dstPath='E:/fmh/Low_light/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train_rsz/train_L/';

imdir = dir([imgPath '*.png']);
ll = length(imdir);
gamma = 1.1 + 0.2*rand(1,ll);
a = -0.3293;
b = 1.1258;
scale = rand(1,ll);
light_scale = 0.5+0.6*rand(1,ll);
flag = rand(1,ll);
%color_tag = rand(1,ll);
color_R = 0.9+0.15*rand(1,ll);
color_G = 0.9+0.2*rand(1,ll);
color_B = 0.9+0.2*rand(1,ll);
%sigma_read = power(10.0, -3.0+1.25*rand(1,ll));
%sigma_shot = power(10.0, -4.0+2.0*rand(1,ll));
sigma_s = 0.016*rand(1,ll)./light_scale;
sigma_c = 0.008*rand(1,ll);

fid=fopen(['.\','val_e.txt'],'a');

for i=1:ll
    fname = imdir(i).name;
    Sname = [fname(1:end-5), 'S.png'];
    img = imread([imgPath imdir(i).name]);
    
    img_01 = im2double(img);    
	% Apply inverse Camera response function
    img_raw = (1 - log(img_01)./b).^(1./a); % beta-gamma inv_crf
    tmp = scale(i)*0.5+0.5;
    I_ = img_raw.*tmp;
	
    % Add color distortion
    R = I_(:,:,1);
    G = I_(:,:,2);
    B = I_(:,:,3);
    R = R.*color_R(i);
    G = G.*color_G(i);
    B = B.*color_B(i);
    img_D = cat(3,R,G,B);
    %delta = sqrt(beta(i)*(0.01/255)^2);
	
	% Avoid Too many bright images
    if light_scale(i)>0.85
        if flag(i)<0.85
            light_scale(i)=light_scale(i)*0.8;
            sigma_s(i) = sigma_s(i)/0.8;
        end
        if light_scale(i)>0.9
            sigma_c(i)=0;
            sigma_s(i)=sigma_s(i)*0.1;
        end
    end
    delta = sigma_s(i)^2.0 + sigma_c(i)^2.0;
    imn = imgAddNoise(img_D, 0, delta);
    [x,y,z] = size(imn);
	
	%Clip
    for j=1:x
        for k=1:y
            for z=1:3
                if(imn(j,k,z)<=0)
                    imn(j,k,z)=0.0001;
                end
                if(imn(j,k,z)>1)
                    imn(j,k,z)=0.9999;
                end
            end
        end
    end
    
    img_re = imn/tmp;
    img_dark = img_re * light_scale(i);
	
	% Use gamma correction to gain non-linear illumination distribution 
    img_LL = im2uint8(img_dark.^(1/gamma(i))); 
    imwrite(img_LL,[dstPath fname(1:end-5) 'L.png'], 'png');
    fprintf(fid,'%s\r\n',[dstPath fname(1:end-5) 'L.png']);
    fprintf(fid,'sigma_s= %.4f ', sqrt(delta));
    fprintf(fid,' light_scale= %.4f\r\n',light_scale(i));

end

fclose(fid);