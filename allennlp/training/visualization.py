from typing import Dict
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

def MR_visualize(predict_result: Dict,
                  serialization_dir: str=None,
                  show_img: bool=False,
                  show_colorbar: bool=True) -> None:
    """
    只输入一个问题和一篇文章，输出一组图
    保存attention和最后预测时概率分布值可视化结果
    """
    if serialization_dir is not None:
        if os.path.exists(serialization_dir) == False:
            os.mkdir(serialization_dir)

    attention_visualize(predict_result, serialization_dir, show_img, show_colorbar)
    vec_visualize(predict_result, serialization_dir, show_img)

def attention_visualize(predict_result: Dict,
                  serialization_dir: str=None,
                  show_img: bool=False,
                  show_colorbar: bool=True) -> None:
    """
    可视化显示模型中间层的“文章-问题”相似度矩阵
    """
    
    cur_time = time.strftime("%Y%m%d_%H%M", time.localtime())
    if serialization_dir is not None:
        cur_time = serialization_dir + '/' + cur_time

    passage = np.array(predict_result['passage_tokens'])

    attention = np.array(predict_result['passage_question_attention'])
    if attention.shape[0] > attention.shape[1]:
        attention = attention.transpose()

    col = np.array(predict_result['passage_tokens'])
    ind = np.array(predict_result['question_tokens'])
    df = pd.DataFrame(attention, columns=col, index=ind)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
    if show_colorbar==True:
        fig.colorbar(cax)
    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_xticklabels([''] + list(df.columns), rotation=90, fontsize=4)
    ax.set_yticklabels([''] + list(df.index), fontsize=4)

    #保存
    savedir = cur_time+'_attention.jpg'
    plt.savefig(savedir, dpi=600)
    if show_img:
        plt.show()


def vec_visualize(predict_result: Dict,
                  serialization_dir: str=None,
                  show_img: bool=False) -> None:
    """
    可视化显示模型最后预测的start，end概率分布值
    """

    passage = np.array(predict_result['passage_tokens'])

    start_logits = np.array(predict_result['span_start_logits'])
    start_probs = np.array(predict_result['span_start_probs'])
    end_logits = np.array(predict_result['span_end_logits'])
    end_probs = np.array(predict_result['span_end_probs'])
    #极值的索引
    sLogit_max = np.argmax(start_logits)
    eLogit_max = np.argmax(end_logits)
    sProb_max = np.argmax(start_probs)
    eProb_max = np.argmax(end_probs)

    cur_time = time.strftime("%Y%m%d_%H%M", time.localtime())
    if serialization_dir is not None:
        cur_time = serialization_dir + '/' + cur_time

    #文章词数
    n_bins = len(passage)

    #Logits图
    plt.figure()
    
    plt.plot(range(n_bins), start_logits, label="start_logits", linewidth=0.8)
    plt.plot(range(n_bins), end_logits, label="end_logits", linewidth=0.8)
    plt.annotate(passage[sLogit_max], xy=(sLogit_max, start_logits[sLogit_max]), xytext=(sLogit_max-15, start_logits[sLogit_max]),
                arrowprops=dict(facecolor='black', headwidth=4, headlength=3, width=0.3),
                )
    plt.annotate(passage[eLogit_max], xy=(eLogit_max, end_logits[eLogit_max]), xytext=(eLogit_max+5, end_logits[eLogit_max]),
            arrowprops=dict(facecolor='black', headwidth=4, headlength=3, width=0.3),
            )        
    bottom, top = plt.ylim()
    ax = plt.gca() 
    ax.spines['bottom'].set_position(('data', bottom))
    plt.xticks(range(n_bins), passage, rotation=90, fontsize=4) #设置spines后xticks设置的属性消失，xticks放在后面

    plt.plot([sLogit_max,sLogit_max],[bottom,start_logits[sLogit_max]], color ='red', linewidth=0.5, linestyle="--")
    plt.plot([eLogit_max,eLogit_max],[bottom,end_logits[eLogit_max]], color ='red', linewidth=0.5, linestyle="--")
    plt.legend()
    #保存
    savedir = cur_time+'_logit.jpg'
    plt.savefig(savedir, dpi=600)
    
    #Probs图
    plt.figure()
    plt.plot(range(n_bins), start_probs, label="start_probabiliy", linewidth=0.8)
    plt.plot(range(n_bins), end_probs, label="end_probabiliy", linewidth=0.8)
    plt.annotate(passage[sProb_max], xy=(sProb_max, start_probs[sProb_max]), xytext=(sProb_max-15, start_probs[sProb_max]),
                arrowprops=dict(facecolor='black', headwidth=4, headlength=3, width=0.3),
                )
    plt.annotate(passage[eProb_max], xy=(eProb_max, end_probs[eProb_max]), xytext=(eProb_max+5, end_probs[eProb_max]),
            arrowprops=dict(facecolor='black', headwidth=4, headlength=3, width=0.3),
            )        
    bottom, top = plt.ylim()
    ax = plt.gca() 
    ax.spines['bottom'].set_position(('data', bottom))
    plt.xticks(range(n_bins), passage, rotation=90, fontsize=4)  #设置spines后xticks设置的属性消失，xticks放在后面

    plt.plot([sProb_max,sProb_max],[bottom,start_probs[sProb_max]], color ='red', linewidth=0.5, linestyle="--")
    plt.plot([eProb_max,eProb_max],[bottom,end_probs[eProb_max]], color ='red', linewidth=0.5, linestyle="--")
    plt.legend()
    #保存
    savedir = cur_time+'_prob.jpg'
    plt.savefig(savedir, dpi=600)

    if show_img:
        plt.show()


def to_predict(model_file, txt_file, question, visual:bool = True, save_dir:str = None, show_img:bool = False)
    from allennlp.predictors import BidafPredictor
    # load model
    if model_file is None:
        bidaf = BidafPredictor.from_path('F:/dl-data/save/origin/model.tar.gz')
    else:
        bidaf = BidafPredictor.from_path(model_file)
    # predict
    if txt_file is None:
        with open("F:\\test.txt", 'r') as f:
    else:
        with open(txt_file, 'r') as f:
            res = bidaf.predict('Who designed the Vince Lombardi Trophy?', f.read())

    if visual:
        if save_dir is None:
            MR_visualize(res, 'F:/zzzzz', show_img=False)
        else:
            MR_visualize(res, save_dir, show_img=False)
