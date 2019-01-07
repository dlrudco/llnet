clear all;
path = 'recon_n';
dList =dir(strcat(path,'/*.jpg')); 
k =length(dList);
patch = gpuArray(uint8(zeros(17,17,3,8370)));
weight = zeros(200,419,3);
temp = ones(17,17,3);
nimg = zeros(200,419,3);

for i=1:1:k
    imgname = sprintf('%d.jpg',i-1);
    patch(:,:,:,i) = imread(strcat(path,'/',imgname));
end
lpatch = gather(patch);
for i = 1:1:62
    for j = 1:1:135
        weight(3*(i-1)+1:3*(i-1)+17,3*(j-1)+1:3*(j-1)+17,:) = weight(3*(i-1)+1:3*(i-1)+17,3*(j-1)+1:3*(j-1)+17,:) + temp;
        nimg(3*(i-1)+1:3*(i-1)+17,3*(j-1)+1:3*(j-1)+17,:) = nimg(3*(i-1)+1:3*(i-1)+17,3*(j-1)+1:3*(j-1)+17,:) + double(lpatch(:,:,:,135*(i-1)+j));
    end
end
nimg = uint8(nimg ./ weight);
imshow(nimg);