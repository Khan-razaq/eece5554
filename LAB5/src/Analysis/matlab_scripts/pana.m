close all;
clear
clc

buildingDir = fullfile('/MATLAB Drive/Examples/R2023b/vision/FeatureBasedPanoramicImageStitchingExample/fifteen');

% Resize images if they are larger than 150 KB
fileExtensions = {'.jpg', '.png', '.jpeg'};

% Loop over each file extension and resize images if they are larger than 150 KB
for ext = fileExtensions
    filePattern = fullfile(buildingDir, ['*', ext{1}]); % ext is a cell array, use ext{1} to extract string
    theFiles = dir(filePattern);
    for k = 1:length(theFiles)
        baseFileName = theFiles(k).name;
        fullFileName = fullfile(buildingDir, baseFileName);
        fileInfo = dir(fullFileName);
        fileSize = fileInfo.bytes;

        if fileSize > 150 * 1024
            I = imread(fullFileName);
            I_resized = imresize(I, 0.75); % Adjust this factor as needed
            imwrite(I_resized, fullFileName);
        end
    end
end

% Now create the imageDatastore for the (possibly resized) images
buildingScene = imageDatastore(buildingDir, 'FileExtensions', {'.jpg', '.jpeg', '.png'});

% Display images to be stitched.
montage(buildingScene.Files)

% Read the first image from the image set.
I = readimage(buildingScene,1);

% Initialize features for I(1)
% Initialize features for I(1)
grayImage = im2gray(I);
[y, x, m] = harris(grayImage, 1000, 'tile', [2 2]);
points = [x, y];
figure
imshow(grayImage)
hold on
temp = cornerPoints(points, "Metric", m);
plot(temp)
[features, points] = extractFeatures(grayImage,points);

% Initialize all the transformations to the identity matrix. Note that the
% projective transformation is used here because the building images are fairly
% close to the camera. For scenes captured from a further distance, you can use
% affine transformations.
numImages = numel(buildingScene.Files);
tforms(numImages) = projtform2d;

% Initialize variable to hold image sizes.
imageSize = zeros(numImages,2);

% Iterate over remaining image pairs
for n = 2:numImages
    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;
        
    % Read I(n).
    I = readimage(buildingScene, n);
    
    % Convert image to grayscale.
    grayImage = im2gray(I);    
    
    % Save image size.
    imageSize(n,:) = size(grayImage);
    
    % Detect and extract SURF features for I(n).
    [y,x,m] = harris(grayImage, 1000, 'tile', [2 2]);
    points = [x, y];
    figure
    imshow(grayImage)
    hold on
    temp = cornerPoints(points, "Metric", m);
    plot(temp)
    [features, points] = extractFeatures(grayImage, points);

    
    % Check if features and featuresPrevious are of the same class
    if ~isa(features, class(featuresPrevious))
        % Convert features to the same class as featuresPrevious
        features = cast(features, class(featuresPrevious));
    end
  
    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);
       
    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);        
    
    % Estimate the transformation between I(n) and I(n-1).
    tforms(n) = estgeotform2d(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    
    % Compute T(1) * T(2) * ... * T(n-1) * T(n).
    tforms(n).A = tforms(n-1).A * tforms(n).A; 
end
     
% Compute the output limits for each transformation.
for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);    
end
avgXLim = mean(xlim, 2);
[~,idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);
     
Tinv = invert(tforms(centerImageIdx));
for i = 1:numel(tforms)    
    tforms(i).A = Tinv.A * tforms(i).A;
end

for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Find the minimum and maximum output limits. 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width 3], 'like', I);

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');  

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:numImages
    
    I = readimage(buildingScene, i);   
   
    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
                  
    % Generate a binary mask.    
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
    
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
end

figure
imshow(panorama)