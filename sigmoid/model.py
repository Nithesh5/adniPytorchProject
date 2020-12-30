from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from sigmoid.data_loader import \
    ADNIDataloaderAllData
import pandas as pd
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam

torch.cuda.empty_cache()

# This code is implemented based on following paper
# https://www.frontiersin.org/articles/10.3389/fnagi.2019.00194/full


if __name__ == "__main__":
    #path to load the labels and images
    csv_file = '../All_Data.csv'
    root_dir = r'C:\StFX\Project\All_Files_Classified\All_Data'

    #dataset
    labels_df = pd.read_csv(csv_file)
    whole_labels_df_AD = labels_df[labels_df.label == "AD"]
    whole_labels_df_CN = labels_df[labels_df.label == "CN"]


    whole_labels_df_AD = whole_labels_df_AD.sample(frac=1, random_state=5)
    whole_labels_df_AD = whole_labels_df_AD.reset_index(drop=True)

    whole_labels_df_CN = whole_labels_df_CN.sample(frac=1, random_state=5)
    whole_labels_df_CN = whole_labels_df_CN.reset_index(drop=True)

    #Train dataset
    labels_df_AD = whole_labels_df_AD.iloc[0:32, :]  #0:320
    labels_df_CN = whole_labels_df_CN.iloc[0:32, :]

    labels_df = labels_df_AD.append(labels_df_CN, ignore_index=True)
    train_labels_df = labels_df.sample(frac=1, random_state=3)
    train_ds = train_labels_df.reset_index(drop=True)

    #Validation dataset
    labels_df_AD = whole_labels_df_AD.iloc[32:42, :] #320:420
    labels_df_CN = whole_labels_df_CN.iloc[32:42, :]
    labels_df = labels_df_AD.append(labels_df_CN, ignore_index=True)
    val_labels_df = labels_df.sample(frac=1, random_state=3)
    val_ds = val_labels_df.reset_index(drop=True)

    #Test dataset
    labels_df_AD = whole_labels_df_AD.iloc[42:47, :] #420:471
    labels_df_CN = whole_labels_df_CN.iloc[42:47, :]
    labels_df = labels_df_AD.append(labels_df_CN, ignore_index=True)
    test_labels_df = labels_df.sample(frac=1, random_state=3)
    test_ds = test_labels_df.reset_index(drop=True)

    print("len of train and val and test")
    print(len(train_ds), len(val_ds), len(test_ds))

    compose = transforms.Compose([
      transforms.ToTensor()
    ])

    train_dataset = ADNIDataloaderAllData(df=train_ds,
                                        root_dir=root_dir,
                                        transform=compose)
    val_dataset = ADNIDataloaderAllData(df=val_ds,
                                        root_dir=root_dir,
                                        transform=compose)

    test_dataset = ADNIDataloaderAllData(df=test_ds,
                                      root_dir=root_dir,
                                      transform=compose)
    train_batch_size = 2
    batch_size = 1

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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
            x = self.sigmoid(x)
            return x

    # checking for device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ADNI_MODEL().to(device)
    print("model defined")

    #Optmizer and loss function

    lr = 1e-4
    wd = 1e-4

    optimizer=Adam(model.parameters(),lr=lr,weight_decay=wd)

    num_epochs=2
    train_count = len(train_ds)
    val_count = len(val_ds)
    test_count = len(test_ds)

    for epoch in range(num_epochs):
        #training on training dataset
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
            loss = F.binary_cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data * images.size(0)

            preds = torch.round(outputs)
            train_accuracy += int(torch.sum(preds.data == labels.data))

        train_accuracy = train_accuracy / train_count
        train_loss = train_loss / train_count

        #To save the model after each epoch
        #torch.save(model.state_dict(), 'best_checkpoint_' + str(epoch) + '.model')

        #Evaluation on testing dataset
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
                loss = F.binary_cross_entropy(outputs, labels)
                val_loss += loss.cpu().data * images.size(0)

                preds = torch.round(outputs)
                val_accuracy += int(torch.sum(preds.data == labels.data))

        val_accuracy = val_accuracy / val_count
        val_loss = val_loss / val_count

        print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) +' Val Loss: ' + str(val_loss) + ' Train Accuracy: ' + str(
            train_accuracy) + ' Val Accuracy: ' + str(val_accuracy))

    print("model saved start")

    #model testing
    model.eval()
    actual_label=[]
    predicted_label=[]

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

            outputs = torch.round(outputs)
            index1 = labels.cpu().data.numpy()
            actual_label.append(index1)
            index2 = outputs.cpu().data.numpy()
            predicted_label.append(index2)
            test_accuracy += int(torch.sum(outputs == labels.data))

    test_accuracy = test_accuracy / test_count

    print(' Test Accuracy: ' + str(test_accuracy))

    from sklearn.metrics import confusion_matrix

    print(actual_label)
    print(predicted_label)

    confusion_matrix = confusion_matrix(actual_label, predicted_label)

    print("Confusion Matrix")
    print(confusion_matrix)

    # Save the best model
    torch.save(model.state_dict(), 'best_checkpoint.model')
    print("model saved")

