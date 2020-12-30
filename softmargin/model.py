from torch.utils.data import DataLoader
from softmargin.data_loader import ADNIDataloaderAllData
import pandas as pd
from torchvision import transforms
import torch.nn as nn
import torch
from torch.optim import Adam

torch.cuda.empty_cache()

if __name__ == "__main__":

    # path to load the labels and images
    csv_file = '../All_Data.csv'
    root_dir = r'C:\StFX\Project\All_Files_Classified\All_Data'
    training_batch_size = 2  # 4
    training_num_workers = 2  # 4
    testing_batch_size = 1
    lr = 1e-4
    wd = 1e-4
    num_epochs = 2

    compose = transforms.Compose([
        transforms.ToTensor()
    ])

    # Train dataset
    labels_df = pd.read_csv(csv_file)
    whole_labels_df_AD = labels_df[labels_df.label == "AD"]
    whole_labels_df_CN = labels_df[labels_df.label == "CN"]

    whole_labels_df_AD = whole_labels_df_AD.sample(frac=1, random_state=5)
    whole_labels_df_AD = whole_labels_df_AD.reset_index(drop=True)

    whole_labels_df_CN = whole_labels_df_CN.sample(frac=1, random_state=5)
    whole_labels_df_CN = whole_labels_df_CN.reset_index(drop=True)

    labels_df_AD = whole_labels_df_AD.iloc[0:320, :]  # 0:320
    labels_df_CN = whole_labels_df_CN.iloc[0:320, :]

    labels_df = labels_df_AD.append(labels_df_CN, ignore_index=True)
    train_labels_df = labels_df.sample(frac=1, random_state=3)
    train_ds = train_labels_df.reset_index(drop=True)

    # Validation dataset
    labels_df_AD = whole_labels_df_AD.iloc[320:420, :]  # 320:420
    labels_df_CN = whole_labels_df_CN.iloc[320:420, :]
    labels_df = labels_df_AD.append(labels_df_CN, ignore_index=True)
    val_labels_df = labels_df.sample(frac=1, random_state=3)
    val_ds = val_labels_df.reset_index(drop=True)

    # Test dataset
    labels_df_AD = whole_labels_df_AD.iloc[420:471, :]  # 420:471
    labels_df_CN = whole_labels_df_CN.iloc[420:471, :]
    labels_df = labels_df_AD.append(labels_df_CN, ignore_index=True)
    test_labels_df = labels_df.sample(frac=1, random_state=3)
    test_ds = test_labels_df.reset_index(drop=True)

    print("len of train and val and test")
    print(len(train_ds), len(val_ds), len(test_ds))

    train_dataset = ADNIDataloaderAllData(df=train_ds,
                                          root_dir=root_dir,
                                          transform=compose)
    val_dataset = ADNIDataloaderAllData(df=val_ds,
                                        root_dir=root_dir,
                                        transform=compose)

    test_dataset = ADNIDataloaderAllData(df=test_ds,
                                         root_dir=root_dir,
                                         transform=compose)

    train_loader = DataLoader(dataset=train_dataset, batch_size=training_batch_size, shuffle=True,
                              num_workers=training_num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=testing_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=testing_batch_size, shuffle=False, num_workers=0)


    class ADNI_MODEL(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.Conv_1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3)
            self.Conv_1_bn = nn.BatchNorm3d(8)
            self.Conv_1_mp = nn.MaxPool3d(2)
            self.Conv_2 = nn.Conv3d(8, 16, 3)
            self.Conv_2_bn = nn.BatchNorm3d(16)
            self.Conv_2_mp = nn.MaxPool3d(3)
            self.Conv_3 = nn.Conv3d(16, 32, 3)
            self.Conv_3_bn = nn.BatchNorm3d(32)
            self.Conv_3_mp = nn.MaxPool3d(2)
            self.Conv_4 = nn.Conv3d(32, 64, 3)
            self.Conv_4_bn = nn.BatchNorm3d(64)
            self.Conv_4_mp = nn.MaxPool3d(3)
            self.relu = nn.ReLU()
            self.dropout1 = nn.Dropout(p=0.4)
            self.dense_1 = nn.Linear(4800, 128)
            self.dropout2 = nn.Dropout(p=0.4)
            self.dense_2 = nn.Linear(128, 1)
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()
            self.softmax = nn.Softmax(dim=1)  # bz applying cross entropy , it has inbuilt softmargin

        def forward(self, x):
            x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
            x = self.Conv_1_mp(x)
            x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
            x = self.Conv_2_mp(x)
            x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
            x = self.Conv_3_mp(x)
            x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
            x = self.Conv_4_mp(x)
            x = x.view(x.size(0), -1)
            x = self.dropout1(x)
            x = self.relu(self.dense_1(x))
            x = self.dropout2(x)
            x = self.dense_2(x)
            x = self.tanh(x)
            return x

    # checking for device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ADNI_MODEL().to(device)
    print("model defined")

    # Optmizer and loss function
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.SoftMarginLoss()

    train_count = len(train_ds)
    val_count = len(val_ds)
    test_count = len(test_ds)
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        # training on training dataset
        model.train()
        train_accuracy = 0.0
        train_loss = 0.0
        val_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.type(torch.FloatTensor)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.reshape(-1)
            outputs = outputs.type(torch.FloatTensor)
            outputs = outputs.cuda()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data * images.size(0)

            rounded_output = []
            for i in outputs.data:
                if i >= 0:
                    preds = 1.0
                else:
                    preds = -1.0
                rounded_output.append(preds)

            train_accuracy += int(torch.sum(preds == labels.data))

        train_accuracy = train_accuracy / train_count
        train_loss = train_loss / train_count

        # torch.save(model.state_dict(), 'best_checkpoint_' + str(epoch) + '.model')

        # Evaluation on testing dataset
        model.eval()

        val_accuracy = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.type(torch.FloatTensor)
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                outputs = model(images)
                outputs = outputs.reshape(-1)
                outputs = outputs.type(torch.FloatTensor)
                outputs = outputs.cuda()
                labels = labels.type(torch.FloatTensor)
                labels = labels.cuda()
                loss = criterion(outputs, labels)
                val_loss += loss.cpu().data * images.size(0)
                rounded_output = []
                for i in outputs.data:
                    if i >= 0:
                        preds = 1.0
                    else:
                        preds = -1.0
                    rounded_output.append(preds)
                rounded_output = torch.FloatTensor(rounded_output)
                rounded_output = rounded_output.cuda()
                val_accuracy += int(torch.sum(rounded_output == labels.data))

        val_accuracy = val_accuracy / val_count
        val_loss = val_loss / val_count

        print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Val Loss: ' + str(
            val_loss) + ' Train Accuracy: ' + str(
            train_accuracy) + ' Val Accuracy: ' + str(val_accuracy))

    print("model saved start")

    # model testing
    model.eval()
    x = []
    y = []

    test_accuracy = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            outputs = outputs.reshape(-1)
            outputs = outputs.type(torch.FloatTensor)
            outputs = outputs.cuda()

            rounded_output = []
            for i in outputs.data:
                if i >= 0:
                    preds = 1.0
                else:
                    preds = -1.0
                rounded_output.append(preds)
            rounded_output = torch.FloatTensor(rounded_output)
            rounded_output = rounded_output.cuda()

            index1 = labels.cpu().data.numpy()
            x.append(index1)
            index2 = rounded_output.cpu().data.numpy()
            y.append(index2)
            test_accuracy += int(torch.sum(rounded_output == labels.data))

    test_accuracy = test_accuracy / test_count

    print(' Test Accuracy: ' + str(test_accuracy))

    from sklearn.metrics import confusion_matrix

    print(x)
    print(y)

    cm = confusion_matrix(x, y)

    print("cm")
    print(cm)

    # Save the best model
    torch.save(model.state_dict(), 'best_checkpoint_2.model')
    print("model saved")
