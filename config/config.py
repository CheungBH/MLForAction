import torch


device = "cuda:0"
print("Using {}".format(device))

X_vector_dim = 36
time_step = 15
y_vector_dim = 5
LSTM_classes = ["null", "FollowThrough", "Standing", "BackSwing", "Downswing"]
LSTM_model = ''
pose_model = ''

input_size_dict = {"inception": 299, "resnet18": 224, "resnet34": 224, "resnet50": 224, "resnet101": 224,
                   "resnet152": 224, "CNN": 224, "LeNet": 28, "mobilenet":224, "shufflenet": 224}

video_path = 'Video/golf/doing/00_Trim.mp4'
webcam_num = 0
CNN_golf_model = 'models/CNN/golf/golf_ske_shufflenet_2019-10-14-08-36-18.pth'
CNN_golf_pre_train_model = 'shufflenet'
CNN_golf_classes = ["Backswing", "FollowThrough", "Standing"]
golf_image_input_size = input_size_dict[CNN_golf_pre_train_model]

golf_static_step = 10


