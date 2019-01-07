%dList =dir('part10/*.jpg'); 
%���� JPG���������� �ƴ� �ٸ� ���� ������ ���� ���,�ݺ������� [dir �Լ�]�� ����ϱ� �ٶ�.
% �ش� ��ɾ�� �̹��� ���ϵ��� �� ã�� �� ����.
% ���� �̹����� �ٸ� ���丮�� ���� ���,  ["���" ".Ȯ���"] ���� ó������
%k =length(dList);
patch = gpuArray(zeros(17,17,3,200));
dark_patch = gpuArray(zeros(17,17,3,200));
noise_patch = gpuArray(zeros(17,17,3,200));
comb_patch = gpuArray(zeros(17,17,3,200));
tic
for i=1:1:k
    i
    read = gpuArray(imread(strcat('part10/',dList(i).name)));
    image = read(:,120:1144,:);
    h_ind = int32(1+ceil(size(image,1)-17)*rand([200,1]));
    w_ind = int32(1+ceil(size(image,2)-17)*rand([200,1]));
    coder.unroll();
    for j=1:1:200
        patch(:,:,:,j) =  uint8(image(h_ind(j):h_ind(j)+16,w_ind(j):w_ind(j)+16,:));
        dark_patch(:,:,:,j) = imadjust(uint8(patch(:,:,:,j)),[0 1], [0 1],3*rand(1,1)+2);
        noise_patch(:,:,:,j) = uint8(imnoise(uint8(patch(:,:,:,j)),'gaussian',0,rand(1,1)*0.01));
        comb_patch(:,:,:,j) = imadjust(uint8(noise_patch(:,:,:,j)),[0 1], [0 1],3*rand(1,1)+2);
    end
    lpatch=uint8(gather(patch));
    dark_lpatch=uint8(gather(dark_patch));
    noise_lpatch=uint8(gather(noise_patch));
    comb_lpatch=uint8(gather(comb_patch));
    for j=1:1:200
        fn0 =  sprintf('%s\\%d.jpg', 'X:\Github\llnet\original', i*200+j);
        fn1 =  sprintf('%s\\%d.jpg', 'X:\Github\llnet\darken', i*200+j);
        fn2 =  sprintf('%s\\%d.jpg', 'X:\Github\llnet\noise', i*200+j);
        fn3 =  sprintf('%s\\%d.jpg', 'X:\Github\llnet\combine', i*200+j);
        imwrite(lpatch(:,:,:,j),fn0);
        imwrite(dark_lpatch(:,:,:,j),fn1);
        imwrite(noise_lpatch(:,:,:,j),fn2);
        imwrite(comb_lpatch(:,:,:,j),fn3);
    end
end
toc