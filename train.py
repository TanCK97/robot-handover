import os

import torch.nn as nn
import torch.nn.functional as F

class HandoverDataset(Dataset):
    """Handover dataset"""

    def __init__(self, img_dir):
        self.img_dir = img_dir

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.annos[idx]["object_detection"]["pred_boxes"]:
            temp = self.annos[idx]["object_detection"]["pred_boxes"][0]
            obj_det = [1]
            temp = np.asarray(temp)
            temp = temp.flatten()

            key_det = self.annos[idx]["keypoint_detection"]["pred_keypoints"][0]
            key_det = np.asarray(key_det)
            key_det = key_det[0:11, 0:2]
            key_det = np.subtract(key_det, temp[0:2])
            key_det = key_det.flatten()

        else:
            obj_det = [-1]
            obj_det = np.asarray(obj_det)

            key_det = self.annos[idx]["keypoint_detection"]["pred_keypoints"][0]
            key_det = np.asarray(key_det)
            key_det = key_det[0:11, 0:2]
            key_det = key_det.flatten()

        if self.annos[idx]["head_pose_estimation"]["predictions"]:
            hp_est = self.annos[idx]["head_pose_estimation"]["predictions"][0]
            hp_est = np.asarray(hp_est)
        else:
            hp_est = np.asarray([-100, -100, -100])

        label = self.annos[idx]["label"]

        anno_list = np.concatenate((obj_det, key_det, hp_est))
        sample = {"annotations": anno_list, "class": label}

        return sample

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        output = self.sigmoid(x)
        return output

def main(args):
    input_size = 26
    output_size = 1

    handover_dataset = HandoverDataset(args.json_path[0])
    handover_train, handover_test = random_split(handover_dataset, (round(0.8*len(handover_dataset)), round(0.2*len(handover_dataset))))

    trainloader = DataLoader(handover_train, batch_size=16, shuffle=True, num_workers=2)
    testloader = DataLoader(handover_test, batch_size=1, shuffle=False, num_workers=2)

    weights_pth = args.weights_path[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=input_size, output_size=output_size)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    epochs = 200

    if args.eval_only:
        do_test(model, device, testloader, weights_pth)
    else:
        do_train(model, device, trainloader, criterion, optimizer, epochs, weights_pth)
        do_test(model, device, testloader, weights_pth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to train feature vector network"
    )
    parser.add_argument(
        "--json-path",
        default="./",
        nargs="+",
        metavar="JSON_PATH",
        help="Path to the json file",
        type=str
    )
    parser.add_argument(
        "--weights-path",
        default="./",
        nargs="+",
        metavar="WEIGHTS_PATH",
        help="Path to the weights file",
        type=str
    )
    parser.add_argument(
        "--eval-only",
        help="set model to evaluate only",
        action='store_true'
    )

    args = parser.parse_args()

    main(args)
