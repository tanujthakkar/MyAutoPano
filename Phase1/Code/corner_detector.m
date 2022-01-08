img = imread('../Data/Train/Set1/1.jpg')
if size(img, 3) > 1
    img = rgb2gray(img);
end
    cimg = cornermetric(img, 'Harris');

cimg = uint8((cimg*127.5) + 127.5);
% min(cimg);
imshow(cimg);