import json
import logging

import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

from src.course.MobileNets.v3.mobilenetv3_02 import mobilenet_v3_large


def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    image_path = "/Users/chendai/Downloads/frames/7_0290.jpg"
    img = Image.open(image_path)
    plt.imshow(img)

    img = data_transform(img)
    img = torch.unsqueeze(img, 0)

    json_path = "/Users/chendai/Documents/workspace/masterai/src/course/MobileNets/v3/class_indices.json"

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model = mobilenet_v3_large(num_classes=2)
    model_weight_path = "model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).cpu()


    # print(f">>> class_indict: {predict_cla}, {class_indict}, {predict[predict_cla].numpy()}")
    print(f">>> predict: {predict[predict_cla].numpy()}, class: {class_indict.get(str(predict_cla.numpy()))}")
    # print(f"class: {class_indict[str(predict_cla)]}, prob: {predict[predict_cla].numpy()}")

    print_res = f"class: {class_indict.get(str(predict_cla.numpy()))}, prob: {predict[predict_cla].numpy()}"
    # print_res = "class: {}   prob: {:.3f}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    # print(f">>> {print_res}")
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3f}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    predict()
