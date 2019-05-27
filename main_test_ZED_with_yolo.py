# -*- coding=utf-8 -*-
# py37

from detect import *
from get_light_type import get_light_type
import cv2

from yzn_judge_light import judge_light


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    # parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    # parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    # parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    # print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode  # 设置为评估模式

    # dataloader = DataLoader(
    #     ImageFolder(opt.image_folder, img_size=opt.img_size),
    #     batch_size=opt.batch_size,
    #     shuffle=False,
    #     num_workers=opt.n_cpu,
    # )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    # detect object from  ZED carmer

    capture = cv2.VideoCapture(0)
    batch_i = 0
    while(True):
        batch_i += 1
        # 获取一帧
        ret, frame = capture.read()
        h, w, d = frame.shape
        frame = frame[0:416, 0:416]  # 双目摄像头只显示其中的一半
        # frame = frame[0:h, 0:int(w / 2)]  # 双目摄像头只显示其中的一半

        transform2 = transforms.Compose([transforms.ToTensor(), ])
        input_imgs = transform2(frame)
        # input_imgs = torch.FloatTensor(frame)
        print("\n", "##" * 66)
        print("Performing object detection:")
        prev_time = time.time()
    # for batch_i, (img_paths, input_imgs) in enumerate(dataloader):  # 遍历一个 bach_size 中的每一张图片
        # Configure input
        # input_imgs = Variable(input_imgs.type(Tensor))
        input_imgs = Variable(input_imgs.type(Tensor).unsqueeze(0))

        # Get detections
        with torch.no_grad():  # ????????????????????
            detections = model(input_imgs)  # 预测一张图片的检测结果
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)  # 非极大抑制（一张图片上的）

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Draw bounding boxes and labels of detections
        frame_copy = frame.copy()
        if detections[0] is not None:
            # Rescale boxes to original image
            # detections = rescale_boxes(detections, opt.img_size, frame.shape[:2])
            # unique_labels = detections[:, -1].cpu().unique()
            # n_cls_preds = len(unique_labels)
            print("Have detect object !!")
            for detection in detections:
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:  # 只输出检测到红绿灯的情况。
                    if cls_pred == 9:
                        print("This is a traffic light")
                        trafficLight = frame[int(y1): int(y2), int(x1):int(x2)]
                        # trafficLight = frame[int(y1) + 2: int(y2) - 2, int(x1) + 2:int(x2)-2]

                        # type = get_light_type(trafficLight_05)  # 去年的方法
                        color, direction, conf = judge_light(trafficLight)  # 今年的新方法，获取红绿灯区域。（目前只考虑了一个红绿灯的情况）

                        frame_copy = cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        font = cv2.FONT_HERSHEY_SIMPLEX

                        # frame_copy = cv2.putText(frame_copy, classes[int(cls_pred)], (x1, y1), font, 1.2, (0, 0, 255), 2)

                        frame_copy = cv2.putText(frame_copy, color + direction + "+" + str(conf), (x1, y1), font, 0.6, (0, 0, 255), 2)

        else:
            print("No object or cant detect")
        cv2.imshow('frame_copy', frame_copy)
        if cv2.waitKey(1) == ord('q'):
            break

