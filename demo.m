clear; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add dependencies
dir_lib_caffe = './fast-rcnn/caffe-fast-rcnn/matlab/caffe/';
dir_lib_frcn = './fast-rcnn/matlab/';
dir_lib_edgebox = './edges/';
dir_lib_ptoolbox = './toolbox/';

addpath(genpath(dir_lib_caffe));
addpath(genpath(dir_lib_frcn));
addpath(genpath(dir_lib_edgebox));
addpath(genpath(dir_lib_ptoolbox));
 
% Set directories
dir_data = './data/';
file_input_img = [dir_data, 'input_img.jpg'];

file_model_edgebox = [dir_lib_edgebox, 'models/forest/modelBsds.mat'];
file_def_frcn = './models/fashion_detector.prototxt';
file_net_frcn = './models/fashion_detector.caffemodel';

dir_results = './results/';
file_output_bbox = [dir_results, 'bbox_img.mat'];

% Set hyper-parameters
flag_visualize = true;
size_img_max = 227;

% Initialize EdgeBox model
model_edgebox = load(file_model_edgebox);
model_edgebox = model_edgebox.model;
model_edgebox.opts.multiscale = 0;
model_edgebox.opts.sharpen = 2;
model_edgebox.opts.nThreads = 4;

opts_edgebox = edgeBoxes;
opts_edgebox.alpha = .65;      % step size of sliding window search
opts_edgebox.beta  = .65;      % nms threshold for object proposals
opts_edgebox.minScore = .01;   % min score of boxes to detect
opts_edgebox.maxBoxes = 1e4;   % max number of boxes to detect

% Initialize fast R-CNN model
tic;
use_gpu = true;
model_frcn = fast_rcnn_load_net(file_def_frcn, file_net_frcn, use_gpu);
disp('### Initialization completed: ');
toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read image
img_cur = imread(file_input_img);
height_img = size(img_cur, 1);
width_img = size(img_cur, 2);
channel_img = size(img_cur, 3);

% Resize image for speed-up
if max([height_img, width_img]) > size_img_max
	ratio = size_img_max ./ max([height_img, width_img]);
	img_resized = imresize(img_cur, ratio);
else
	ratio = 1;
	img_resized = img_cur;
end

% Generate object proposals using EdgeBox
tic;
try
	proposals_cur = edgeBoxes(img_resized, model_edgebox, opts_edgebox);
catch
	continue;
	warning('!!! error generating object proposals');
end
proposals_cur = proposals_cur(:, 1:4); % only keep coordinates
proposals_cur(:, 3:4) = proposals_cur(:, 1:2) + proposals_cur(:, 3:4); % convert [x, y, w, h] to [x1, y1, x2, y2]
proposals_cur = single(proposals_cur) + 1; % account for the 0-based indexing in EdgeBox
disp('### EdgeBox completed : ');
toc;

% Sanity check proposals 
if isempty(proposals_cur)
	continue;
	warning('!!! no proposal generated');
end

% Detecting clothes using fast R-CNN 
tic;
try
	bbox_pred = fast_rcnn_im_detect(model_frcn, img_resized, proposals_cur);
catch
	continue;
	warning('!!! error running fast R-CNN');
end
disp('### Fast R-CNN completed: ');
toc;

% Get predicted bounding boxes w.r.t resized image
bbox_pred = bbox_pred{1};

% Transform bounding box back into original coordinates w.r.t input image
bbox_img(:, 1:4) = bbox_pred(:, 1:4) ./ ratio;
bbox_img(:, 5) = bbox_pred(:, 5);

% Visualize the top-2 detection results
if flag_visualize
	figure(1);
	subplot(1, 2, 1); showboxes(img_cur, bbox_img(1, :));
	subplot(1, 2, 2); showboxes(img_cur, bbox_img(2, :));
end

% Save bounding box outputs
save(file_output_bbox, 'img_cur', 'bbox_img', '-v7.3');
