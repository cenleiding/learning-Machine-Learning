# svm.SVC 参数解析
 注意 scikit-learn里的svm都使用'hinge'损失函数

## Parameters：
* 惩罚系数C：惩罚系数，默认=1.0 表示软间隔承受程度。∞则为硬间隔。

* 核函数 kernel：

	linear：线性核
	
	poly ： 多项式核 K(x,z)=（γx*z+r)^d
	
	rbf：高斯核(默认) K(x,z)=exp(−γ||x−z||^2) 
	
	sigmoid：sigmoid核 K(x,z)=tanh(γx∙z+r)
	
* 核函数参数degree：

	用于'poly'核：表示多项式的d，多项式次数，默认3
	
* 核函数参数gamma：
	
	'poly'核：表示多项式的γ
	
	'rbf'核: 表示高斯核的γ
	
	'sigmoid'核：同样表示γ
	
	默认为'auto',1/n_features，即将弃用
	
	即将改为默认'scale'，1/(n_features * X.std())

* 核函数参数coef0：

	'poly'核：表示多项式的r
	
	'sigmoid'核：表示r
	
	默认为0
	
* 样本权重class_weight：

	防止样本分布不均。
	
	默认为'None'。
	
	'balanced': n_samples / (n_classes * np.bincount(y))
	
	'自定义'：如{1：100：1000}
	
* 分类决策decision_function_shape：

	'ovo': one vs one	
	
	'ovr', 默认，：one-vs-rest
	
* 缓存大小cache_size:

	in(MB)
	
* 概率probability : 

	默认为False,如果后续要能显示概率，则改为True,但会降低运算速度。
	
## Attributes：
	
* support_ : array-like, shape = [n_SV]

	Indices of support vectors.
	
* support_vectors_ : array-like, shape = [n_SV, n_features]
	
	Support vectors.
	
## Methods：
	
* decision_function(X)：计算样本离超平面的距离

* fit(X, y, sample_weight=None)：拟合

* predict(X)：预测分类，+1 or -1

* predict_log_proba:计算log概率，probability必须为True

* predict_proba:计算概率，probability必须为True

* score(X, y, sample_weight=None)：计算预测得分