
from django.shortcuts import render

# Create your views here.

from joblib import load
import torch
import torch.nn as nn
from .DNN import dense_model
from .CNN import CNN_model
model = load('model.joblib')
scaler=load('scaler.joblib')

def home(request):
   
    return render(request, 'home.html')
def res(request):
    age=int(request.GET['age'])
    gen=int(request.GET['gender'])
    hdi=int(request.GET['heartdisease'])
    hpt=int(request.GET['hypertension'])
    mar=int(request.GET['maritalstatus'])
    wor=int(request.GET['worktype'])
    bmi=float(request.GET['bmi'])
    smoke=int(request.GET['smoking'])
    res=int(request.GET['residence'])
    glu=float(request.GET['glu'])
    sc=scaler.transform([[age,glu,bmi]])
    data=[[gen,sc[0][0],hpt,hdi,mar,wor,res,sc[0][1],sc[0][2],smoke]]
    lr_probs=model.predict_proba(data)[:, 1]
    pred1=model.predict(data)
    data1 = torch.tensor(data, dtype=torch.float32)
    test_outputs = dense_model(data1)
    DNN_probs= torch.softmax(test_outputs, dim=1)[:, 1].detach().numpy()
    test_preds = torch.argmax(test_outputs, dim=1)
    test_outputs_CNN = CNN_model(data1)
    CNN_probs = torch.softmax(test_outputs_CNN, dim=1)[:, 1].detach().numpy()
    test_preds_CNN = torch.argmax(test_outputs_CNN, dim=1)
    if lr_probs > 0.3:
        if 0.4 <DNN_probs < 0.6:
            final_pred = int(CNN_probs > 0.5)
        else:
            final_pred = int(DNN_probs > 0.5)
    else:
        final_pred = 0  
    if final_pred==0:
        hs=0
    else:
        hs=1
    return render(request,'res.html',{'pr':hs})
    