import json
import logging
import os.path
import sys

import torch
from torch import nn, optim
from torchvision import transforms, datasets
from tqdm import tqdm

from src.course.MobileNets.v3.mobilenetv3_02 import MobileNetV3, mobilenet_v3_large


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device} device.")

    batch_size = 16
    epochs = 100

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../datasets/"))
    data_root = "/Users/chendai/Documents/workspace/datasets/"
    image_path = os.path.join(data_root, "finger")

    # print(">>> image path:", image_path)
    assert os.path.exists(image_path), "Image path does not exist!"
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transforms["train"])
    train_num = len(train_dataset)

    # print(train_dataset)
    cls_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in cls_list.items())

    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    logging.info(f"Using {nw} dataloader workers every process.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
    )

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transforms["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=batch_size,
        num_workers=nw,
    )

    logging.info(f"Using {train_num} images for training, {val_num} images for validation.")

    net = mobilenet_v3_large(num_classes=2)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0005)

    best_acc = 0.0
    save_path = "model.pth"
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs.data, 1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num
        logging.info(f"[epoch {epoch + 1}] train_loss: {running_loss / train_steps}, val_accuracy: {val_accurate}")

        print(f">>> val_accurate: {val_accurate}, best_acc: {best_acc}")
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    logging.info(f"Finished training")


if __name__ == '__main__':
    train()