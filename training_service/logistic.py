"""Deterministic dependency-free binary logistic classifier."""

from __future__ import annotations

import math


def _sigmoid(value):
    if value >= 0:
        inverse=math.exp(-value); return 1/(1+inverse)
    exponential=math.exp(value); return exponential/(1+exponential)


def fit(rows,labels,*,iterations=600,learning_rate=.08,l2=.002):
    if not rows or len(rows)!=len(labels): raise ValueError("invalid training rows")
    width=len(rows[0])
    if width<1 or any(len(row)!=width for row in rows): raise ValueError("invalid training width")
    means=[sum(row[index] for row in rows)/len(rows) for index in range(width)]
    scales=[]
    for index,mean in enumerate(means):
        variance=sum((row[index]-mean)**2 for row in rows)/len(rows)
        scales.append(max(math.sqrt(variance),1e-6))
    normalized=[[(value-means[index])/scales[index] for index,value in enumerate(row)] for row in rows]
    weights=[0.0]*width; bias=0.0
    for _ in range(iterations):
        weight_gradient=[0.0]*width; bias_gradient=0.0
        for row,label in zip(normalized,labels):
            error=_sigmoid(sum(w*x for w,x in zip(weights,row))+bias)-label
            for index,value in enumerate(row): weight_gradient[index]+=error*value
            bias_gradient+=error
        count=len(rows)
        for index in range(width):
            weights[index]-=learning_rate*(weight_gradient[index]/count+l2*weights[index])
        bias-=learning_rate*bias_gradient/count
    return {"means":means,"scales":scales,"weights":weights,"bias":bias}


def probability(model,row):
    normalized=[(value-model["means"][index])/model["scales"][index] for index,value in enumerate(row)]
    return _sigmoid(sum(w*x for w,x in zip(model["weights"],normalized))+model["bias"])


def metrics(model,rows,labels):
    predictions=[int(probability(model,row)>=.5) for row in rows]
    tp=sum(p==1 and y==1 for p,y in zip(predictions,labels))
    tn=sum(p==0 and y==0 for p,y in zip(predictions,labels))
    fp=sum(p==1 and y==0 for p,y in zip(predictions,labels))
    fn=sum(p==0 and y==1 for p,y in zip(predictions,labels))
    accuracy=(tp+tn)/len(labels)
    recall_positive=tp/(tp+fn) if tp+fn else 0.0
    recall_negative=tn/(tn+fp) if tn+fp else 0.0
    return {"accuracy":round(accuracy,6),
            "balanced_accuracy":round((recall_positive+recall_negative)/2,6),
            "positive_recall":round(recall_positive,6),"negative_recall":round(recall_negative,6),
            "test_samples":len(labels)}
