from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix,mean_squared_error
import numpy as np


def acc(y_pred,y_true):
    f1 = f1_score(y_pred,y_true,average='macro')
    precision = precision_score(y_pred,y_true,average='macro')
    recall = recall_score(y_pred,y_true,average='macro')
    accuracy = sum(np.array(y_pred)==np.array(y_true))/(0.00+len(y_pred))
    C=confusion_matrix(y_true, y_pred)
    return accuracy,f1,precision,recall,C


def result_show_acc(logging,loss,acc,flag = 'Test!'):
    print('-------------------------------------------------------------')
    print(flag)
    print('loss:%.3f,loss_sent:%.3f,loss_dis:%.3f,mae:%.4f,rmse:%.4f,acc:%.3f,f1:%.3f,precision:%.3f,recall:%.3f' % 
        (loss[0],loss[1],loss[2],loss[3],loss[4],acc[0],acc[1],acc[2],acc[3]))
    logging.info('-------------------------------------------------------------')
    logging.info(flag)
    logging.info('loss:%.3f,loss_sent:%.3f,loss_dis:%.3f,mae:%.4f,rmse:%.4f,acc:%.3f,f1:%.3f,precision:%.3f,recall:%.3f' % 
        (loss[0],loss[1],loss[2],loss[3],loss[4],acc[0],acc[1],acc[2],acc[3]))
    
