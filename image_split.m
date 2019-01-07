clear all;
path = 'test_c';
dList =dir(strcat(path,'/*.jpg'));
k =length(dList);
patch = gpuArray(uint8(zeros(17,17,3,8370)));


read = gpuArray(imread('combine_car.jpg'));

image = read;

for i = 1:1:62
    for j = 1:1:135
        patch(:,:,:,135*(i-1)+j)  = uint8(image(3*(i-1)+1:3*(i-1)+17,3*(j-1)+1:3*(j-1)+17,:));
    end
end
lpatch = uint8(gather(patch));
for i = 1:1:8370
    if rem(i,1000) == 0
        i
    end
    fn = sprintf('%s\\%s\\%d.jpg', 'X:\Github\llnet',path, i);
    imwrite(lpatch(:,:,:,i),fn);
end

