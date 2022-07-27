import torch
from math import exp
import numpy as np
from train_utils import fun

with open('parameters/model.m','rb') as f:
    model=torch.load(f)
    f.close()
with open('parameters/train_data.t','rb') as f:
    train_data=torch.load(f)
    f.close()
with open('parameters/fake_data.t','rb') as f:
    fake_data=torch.load(f)
    f.close()

correct = 0
total = 0

run_data=[(torch.cat([train_data,fake_data])[i],1 if i<train_data.shape[0] else 0)
    for i in range(train_data.shape[0]+fake_data.shape[0])]
train_loader=torch.utils.data.DataLoader(run_data,batch_size=5,shuffle=True)

with torch.no_grad():
    for data, labels in train_loader:
        batch_size=data.shape[0]
        outputs = model(data.view(data.shape[0], -1))
        index, predicted = torch.max(outputs, dim=1)
#        for i in range(batch_size):
#            print('Label',predicted[i].item(),':',exp(outputs[i][1].item())*100)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
        
print("Train Accuracy: %f" % (correct / total))

testdata,filenames=fun.ReadFile('data/test')
testdata=fun.pretreatment(testdata)

print(testdata.shape)
print(train_data.shape)

with torch.no_grad():
    for i in range(len(filenames)):
        data=testdata[i]
        outputs=model(data.view(1,-1))
        index, predicted = torch.max(outputs, dim=1)
        print(filenames[i],predicted[0].item(),':',exp(outputs[0][1].item())*100)