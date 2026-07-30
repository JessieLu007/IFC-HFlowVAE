import os 
import math
import time 
import random 
import pickle
import shutil
import numpy as np 
import pandas as pd
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler 

from imblearn.metrics import geometric_mean_score


def split_and_fill_nans(data_loc, imputer, cat_idxs, int_idxs, test_size=0.3, seed=42):

    #if (len(cat_idxs)==0 or len(int_idxs)==0):
        #raise Exception("You forget to tell us the indexes of categorical attributes and attributes with integer values!")

    train_raw = pd.read_csv(data_loc)
    cols = train_raw.columns
    X = train_raw[cols[0:-1]]
    y = train_raw[cols[-1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    
    train = X_train.copy()
    test = X_test.copy()

    if (imputer == "SimpleImputer"):
        imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        train = imp.fit_transform(train)
        test = imp.transform(test)
    elif (imputer == "IterativeImputer"):
        imp = IterativeImputer(max_iter=10, random_state=seed)
        train = imp.fit_transform(train)
        test = imp.transform(test)
    elif (imputer == "KNNImputer"):
        imp = KNNImputer(n_neighbors=1)
        train = imp.fit_transform(train)
        test = imp.transform(test)
    else:
        raise Exception("Invalid imputer, please check!")
    
    train = pd.DataFrame(train, columns=cols[:-1])
    test = pd.DataFrame(test, columns=cols[:-1])
    
    # change back to integers while the filling nan can introduce float numbers
    if(len(cat_idxs)>0):
        for i in cat_idxs:
            cat_row = train[cols[i]]
            unique_v = list(set(X_train[cols[i]]))
            unique_v = [x for x in unique_v if str(x) != 'nan']
            for index, row in cat_row.items():
                if(not int(row) == row):
                    nearest = [abs(x-row) for x in unique_v]
                    near_ids = nearest.index(min(nearest))
                    near_v = unique_v[near_ids]
                    train.loc[index, cols[i]] = near_v
        
        for i in cat_idxs:
            cat_row = test[cols[i]]
            unique_v = list(set(X_test[cols[i]]))
            unique_v = [x for x in unique_v if str(x) != 'nan']
            for index, row in cat_row.items():
                if(not int(row) == row):
                    nearest = [abs(x-row) for x in unique_v]
                    near_ids = nearest.index(min(nearest))
                    near_v = unique_v[near_ids]
                    test.loc[index, cols[i]] = near_v

    
    y_train = pd.DataFrame(y_train, columns=[cols[-1]])
    y_test = pd.DataFrame(y_test, columns=[cols[-1]])
    y_train.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    # turn attribute values back to integers
    if(len(int_idxs)>0):
        for i in int_idxs:
            train[cols[i]] = train[cols[i]].apply(lambda x: round(x, 0))
            test[cols[i]] = test[cols[i]].apply(lambda x: round(x,  0))

    whole = list(range(X_train.shape[1]))
    float_idxs = list(set(whole)-set(cat_idxs)-set(int_idxs))
    for i in float_idxs:
        train[cols[i]] = train[cols[i]].apply(lambda x: round(x, 2))
        test[cols[i]] = test[cols[i]].apply(lambda x: round(x, 2))

    return train, test, y_train,  y_test, cols



def  encode_and_scale(X_train, X_test, cols, encoder, scaler, cat_idxs, output_distribution=None):

    cat_cols = cols[cat_idxs]

    if(encoder == 'onehot'):
        enc = ce.OneHotEncoder(cols=cat_cols)
        X_train = enc.fit_transform(X_train)
        X_test = enc.transform(X_test)
    elif(encoder == 'binary'):
        enc = ce.BinaryEncoder(cols=cat_cols, )
        X_train = enc.fit_transform(X_train)
        X_test = enc.transform(X_test)
    else:
        raise Exception("Invalid encoder, please check!")
    
    encoded_cols = X_train.columns
    
    if(scaler == 'MinMaxScaler'):
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif(scaler == 'QuantileTransformer'):
        if(output_distribution is None):
            scaler = QuantileTransformer()
        else:
            scaler = QuantileTransformer(output_distribution=output_distribution)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        raise Exception("Invalid scaler, please check!") 
    
    X_train = pd.DataFrame(X_train, columns=encoded_cols)
    X_test = pd.DataFrame(X_test, columns=encoded_cols)
    
    return X_train, X_test, enc, scaler


def inverse_transform (data, cols, encoder, scaler, int_idxs ): 
   
    encoded_cols = data.columns
    
    scale_restore = pd.DataFrame(scaler.inverse_transform(data),  columns=encoded_cols)

    
    if (isinstance(encoder, ce.binary.BinaryEncoder)):
        for item in encoder.mapping:
            #attr = item['col']
            mapping = item['mapping'].drop([-1, -2], axis=0)
            attrs = mapping.columns
            for i, row in scale_restore[attrs].iterrows():
                list_row = list(row)
                first_ids = encoded_cols.get_loc(key=attrs[0])
                for attr in attrs:
                    ids = encoded_cols.get_loc(key=attr)
                    if(row[attr]>=0.5):
                        scale_restore.iat[i, ids] = 1
                    elif(row[attr]<0.5):
                        scale_restore.iat[i, ids] = 0

                num = list(scale_restore[attrs].loc[i,].to_numpy())
                vals = mapping.to_numpy().tolist()
                if(num not in vals):
                    dist_list = [math.dist(list_row, val) for val in vals]
                    ids = dist_list.index(min(dist_list))
                    for j in range(0, len(vals[0])):
                        scale_restore.iat[i, first_ids+j] = vals[ids][j]      
        restore = encoder.inverse_transform(scale_restore)
        restore = pd.DataFrame(restore, columns=cols[:-1])
    elif (isinstance(encoder, ce.one_hot.OneHotEncoder)):
        for item in encoder.mapping:
            mapping = item['mapping'].drop([-1, -2], axis=0)
            attrs = mapping.columns

            for i, row in scale_restore[attrs].iterrows():
                list_row = list(row)
                first_ids = encoded_cols.get_loc(key=attrs[0])
                for attr in attrs:
                    ids = encoded_cols.get_loc(key=attr)
                    if(row[attr]>=0.5):
                        scale_restore.iat[i, ids] = 1
                    elif(row[attr]<0.5):
                        scale_restore.iat[i, ids] = 0
                num = list(scale_restore[attrs].loc[i,].to_numpy())
                vals = mapping.to_numpy().tolist()
                if(num not in vals):
                    max_ids = num.index(max(num))
                    if(len(set(num))==1):
                        max_ids = random.randint(0, len(num)-1)
                    for j in range(0, len(vals[0])):
                        scale_restore.iat[i, first_ids+j] = 0
                    scale_restore.iat[i, first_ids+max_ids] = 1
        restore = encoder.inverse_transform(scale_restore)
        restore = pd.DataFrame(restore, columns=cols[:-1])
    else:
        if(isinstance(encoder, ce.MEstimateEncoder)):
            for item in encoder.mapping.items():
                attr = item[0] 
                mapping = item[1] 
                mapped_values = []
                length = len(mapping.index)-1
                for i in mapping.index:
                    mapped_values.append(round(mapping[i], 6))

                
                trans_mapping = pd.Series(data=mapping.index, index=mapped_values)
                trans_mapping.drop(mapped_values[length-1:length+1], inplace=True)

                for index, value in scale_restore[attr].items():
                    value = round(value, 6)
                    if (value in mapped_values):
                        scale_restore[attr][index] = trans_mapping[value]
                    else:
                        gaps = [round(abs(item - value),6)+random.uniform(0, 0.00002) for item in trans_mapping.index]
                        trace_back = pd.Series(data=trans_mapping.values, index=gaps)
                        min_gaps = min(gaps)
                        scale_restore[attr][index] = trace_back[min_gaps]
            restore = pd.DataFrame(restore, columns=cols[:-1])
        else:
            restore = scale_restore

    for i in int_idxs:
        restore[cols[i]] = restore[cols[i]].apply(lambda x: round(x, 0))

    return restore


def list_subtraction(attr_len, sub_list):
    '''
    get the ordered list of a larger one substract a smaller one
    not be used for now
    '''
    full_list = list(range(0, attr_len))
    sub = list(set(full_list) - set(sub_list))
    sub.sort()
    return sub


# 逆向处理分类属性，单独处理onehot编码方法， 要使得同一个属性的子属性中有且只有一个值为1
def inverse_onehot_categorical(data, original_cols, encoder, cat_idxs): 
    '''
    data: 分类属性编码后的数据
    original_cols: 编码之前的属性名称，不包括标签属性
    encoder: 之前预处理时训练后得到的分类器
    cat_attrs: 要处理的分类属性
    '''
    encoded_cols = data.columns
    cat_attrs = original_cols[cat_idxs]

    for attr in cat_attrs:
        encoded_attrs = [] # 保存当前属性编码后得到的分属性
        for col in encoded_cols:
            if(col.find(attr) != -1):
                encoded_attrs.append(col)
        # 处理当前属性下的属性值        
        for index, row in data[encoded_attrs].iterrows():
            print('test')

    #data[col] = data[col].apply(lambda x: 1 if x>0.5 else 0)


# 获取编码后的子属性名称，用于后续处理，这个有问题
def return_encoded_attrs_dict(data, cols, cat_idxs):
    encoded_cols = data.columns
    cat_attrs = cols[cat_idxs]
    attrs_dict = []
    for attr in cat_attrs:
        encoded_attrs = []
        for col in encoded_cols:
            if (col.find(attr)!= -1):
                encoded_attrs.append(col)
        attrs_dict.append(encoded_attrs)
    return attrs_dict


def inverse_onehot(data, o_cols, cat_idxs):
    '''
    o_cols: 编码前的属性
    data: 编码分类属性后的数据
    cat_idxs: 编码前分类属性的下标
    '''
    cols = data.columns
    encoded_attrs = []
    cat_attrs = o_cols[cat_idxs]
    for attr in cat_attrs:
        attrs = []
        for col in cols:
            if(col.find(attr) != -1):
                attrs.append(col)
        encoded_attrs.append(attrs)

    print(encoded_attrs)

    for attrs in encoded_attrs:
        for i, row in data[attrs].iterrows():
            o_value = max(row)
            positions = np.where(row == o_value)
            max_len = len(positions[0])
            index_max = random.choice(positions[0])
            data.at[i, attrs] = 0
            data.at[i, attrs[index_max]] = 1
    return data

# ===================================分类结果测试函数=============================================================================
# RandomForestClassifier(random_state=42), KNeighborsClassifier(n_neighbors=5),MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1000)
# LogisticRegression(random_state=21),DecisionTreeClassifier(),AdaBoostClassifier()
def classification_test(X_train, y_train, X_test, y_test, loc, save_name):
    classifiers = [SVC(kernel='linear',probability=True),MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1000),RandomForestClassifier(random_state=42)]
    classes = []
    test_scores = []
    class_reports = []
    matrixs = []
    aucs = [] # save roc_auc_score
    g_means = []

    for cls in classifiers:
        y_test = np.ravel(y_test)
        start_time = time.time()
        cls.fit(X_train, np.ravel(y_train))
        print(cls,cls.score(X_test, y_test))
        classes.append(cls)
        test_scores.append(cls.score(X_test, y_test))
        y_pred = cls.predict(X_test)
        y_proba = cls.predict_proba(X_test)[::,1]
        auc = roc_auc_score(y_test, y_proba)
        aucs.append(auc)
        g_mean = geometric_mean_score(y_test, y_pred)
        g_means.append(g_mean)
        class_report = classification_report(y_test, y_pred, digits=3)
        class_reports.append(class_report)
        matrix = confusion_matrix(y_test, y_pred)
        matrixs.append(matrix)
        print(confusion_matrix(y_test, y_pred))
        print(class_report)
        print(f"geometric mean is {g_mean:.3f}")
        print(f"AUC value {auc:.3f}")
        #print('Executing time: %s' % (time.time()-start_time))
        print('----------------------------------------------------------------------------------')

    save_classification_results(classes, test_scores, aucs, g_means, class_reports, matrixs, loc, save_name)
    return


# 保存分类结果用于以后分析
def save_classification_results(classes, test_scores, aucs, g_means, class_reports,matrixes,loc, save_name):
    for i in range(len(classes)):
        clss = classes[i]
        test_score = test_scores[i]
        auc = aucs[i]
        g_mean = g_means[i]
        report = class_reports[i]
        matrix = matrixes[i]

        with open(loc+save_name, 'a') as f:
            f.write(clss.__class__.__name__+' ')
            f.write(report)
            f.write('test score: '+str(test_score)+'\n')
            f.write('G-Mean value: '+str(g_mean)+'\n')
            f.write('AUC value: '+str(auc)+'\n')
            f.write('TN '+str(matrix[0][0])+' ')
            f.write('FN '+str(matrix[0][1])+'\n')
            f.write('FP '+str(matrix[1][0])+' ')
            f.write('TP '+str(matrix[1][1])+' ')
            f.write('\n')
            f.close()
    with open(loc+save_name, 'a') as f:
        f.write('='*100+'\n')
        f.close()
    print('Classification results have been saved to '+loc+save_name)

# 自定义缺失值补全策略，即用已有的属性值来辅助填充缺失值，应该是很原始的方法，比不过软件包里的很多实现
# ===============================================================================================================================================================
# 获取每一列属性的取值，并以字典的方式返回，从代码可以看出所取的并非是unique的，而是照单全收的
def get_values(data):
    attr_values = {}
    for attr, values in data.items():
        vals = []
        for val in values:
            if(val != val):
                pass
            else:
                vals.append(val)
        attr_values[attr] = vals 
    return attr_values
# ===============================================================================================================================================================

# 在训练数据中挑出某一类样本
def pick_one_class(X, y, tag=1):
    label = y.columns[0]
    if(tag==1):
        index =y[y[label]==1].index.tolist()
        chosen = X.iloc[index].copy()
    if(tag==0):
        index = y[y[label]==0].index.tolist()
        chosen = X.iloc[index].copy()
    return chosen, index


# 删除某目录下的所有文件
def deldir(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename) 
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))

# 删除目录下的空文件夹
def delete_empty_folders(root):
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            if not os.listdir(full_path):
                os.rmdir(full_path)

# 绘制生成数据和原数据的各个属性的STD值
import matplotlib.pyplot as plt
def plot_std_comparison(pos_train, generated, name):
    # 绘制生成样本各个属性的均值和方差
    o_stds = pd.DataFrame(pos_train.std())
    o_means = pd.DataFrame(pos_train.mean())

    o_std_vals = []
    for _, row in o_stds.iterrows():
        o_std_vals.append(row[0])
    o_mean_vals = []
    for _, row in o_means.iterrows():
        o_mean_vals.append(row[0])
    
    stds = pd.DataFrame(generated.std())
    means = pd.DataFrame(generated.mean())
    std_vals = []
    for _, row in stds.iterrows():
        std_vals.append(row[0])
    mean_vals = []
    for _, row in means.iterrows():
        mean_vals.append(row[0])
    
    plt.plot(o_std_vals, label='original')
    plt.plot(std_vals, label=name)
    plt.legend()
    #plt.savefig(save_loc+name+'_std_comparison.pdf')
    plt.show()
    return o_std_vals, std_vals


# 补充新的正例样本到训练集中
def add_pos_samples(X, y, sup_X, drop=False, label='label'):
    train = pd.concat([X, y], axis=1)
    cols = train.columns
    
    if(drop):
        '''
        如果drop, 就表示采用替换的策略，用生成样本替换原先的样本；
        这样最后得到的样本数目不变
        '''
        pos_train_index = y[y[label]==1].index.tolist()
        drop_index = random.sample(pos_train_index, sup_X.shape[0])
        train.drop(drop_index, axis=0, inplace=True)
    
    sup_y = pd.DataFrame(np.ones(len(sup_X)), columns=[label])

    sup_new = pd.concat([sup_X, sup_y], axis=1)
    train_new = pd.concat([train, sup_new], axis=0)
    train_new = shuffle(train_new, random_state=21)
    train_new = train_new.reset_index(drop=True)
    X_new = train_new[cols[:-1]]
    y_new = train_new[cols[-1]]
    
    return X_new, y_new


# 读取pickle保存的变量
def read_from_local(file_name):
    open_file = open(file_name, 'rb')
    variable = pickle.load(open_file)
    open_file.close()
    return variable

# 用于将变量保存到本地的函数
def dump_to_local(var, var_name, location):
    open_file = open(location+var_name,'wb')
    pickle.dump(var, open_file)
    open_file.close()


# 保存用于绘制PR图的数据
def save_precision_recall(loc, precision, recall, ap):
    name = loc+'prc_data.txt'
    with open(name, 'a') as f:
        for p in precision:
            f.write(str(p)+ " ")
        f.write("\n")
        for r in recall:
            f.write(str(r)+ " ")
        f.write("\n")
        f.write(str(ap)+"\n")
        f.close()
    return 0


#=================================================================================================================
# https://www.imranabdullah.com/2019-06-01/Drawing-multiple-ROC-Curves-in-a-single-plot 
#=================================================================================================================
from sklearn.metrics import roc_curve
def plot_and_save_ROC(X, y, X_vae, y_vae, X_smote, y_smote, X_valid, y_valid, save_loc):
    clses = [SVC(kernel='linear',probability=True), RandomForestClassifier(),LogisticRegression(max_iter=1500),
        MLPClassifier(hidden_layer_sizes=(16, 32), max_iter=1000), KNeighborsClassifier()]
    try:
        os.makedirs(save_loc+'ROC_Curves\\')
    except:
        deldir(save_loc+'ROC_Curves\\')
    fig_save = save_loc+'ROC_Curves\\'
    print('The save location of ROC_Curves are: '+fig_save)

    for cls in clses:
        y_copy = np.ravel(y)
        y_vae_copy = np.ravel(y_vae) 
        y_smote_copy = np.ravel(y_smote)
        train_Xs = [X, X_vae, X_smote]
        train_ys = [y_copy, y_vae_copy, y_smote_copy]
        names = ['original', 'vae', 'smote']
        plot_cols = ['datas', 'fpr', 'tpr', 'auc']
        result_table = pd.DataFrame(columns=plot_cols)
        i = 0
        for X, y in zip(train_Xs, train_ys):
            model = cls.fit(X, y)
            yproba = model.predict_proba(X_valid)[::,1]
            fpr, tpr, _ = roc_curve(y_valid, yproba)
            auc = roc_auc_score(y_valid, yproba)
            result_table = pd.concat([result_table, pd.DataFrame([[names[i], fpr, tpr, auc]], columns=plot_cols)], axis=0, ignore_index=True)
            i += 1
        result_table.set_index('datas', inplace=True)
        # Draw the curves
        colors = ['b','r','g']
        c = 0
        fig = plt.figure(figsize=(8,6))
        for i in result_table.index: 
            plt.plot(result_table.loc[i]['fpr'],
                result_table.loc[i]['tpr'], color=colors[c],
                label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
            c += 1
        plt.plot([0,1], [0,1], color='black', linestyle='--')
        plt.xticks(np.arange(0.0, 1.1, step=0.1))
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.yticks(np.arange(0.0, 1.1, step=0.1))
        plt.ylabel('True Positive Rate', fontsize=15)
        plt.title(cls.__class__.__name__+' ROC_Curves', fontweight='bold', fontsize=15)
        plt.legend(prop={'size':13}, loc='lower right')
        plt.savefig(fig_save+cls.__class__.__name__+'.pdf')
        plt.show()


from sklearn.metrics import PrecisionRecallDisplay
def plot_and_save_PRC(X, y, X_vae, y_vae, X_smote, y_smote, X_valid, y_valid, save_loc):
    clses = [SVC(kernel='linear', probability=True), DecisionTreeClassifier(random_state=42), 
            MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=1000), RandomForestClassifier(random_state=42)]
    vae_clses = [SVC(kernel='linear', probability=True), DecisionTreeClassifier(random_state=42), 
            MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=1000), RandomForestClassifier(random_state=42)]
    smote_clses = [SVC(kernel='linear', probability=True), DecisionTreeClassifier(random_state=42), 
            MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=1000), RandomForestClassifier(random_state=42)]

    for i in range(4):
        cls = clses[i]
        cls_vae = vae_clses[i]
        cls_smote = smote_clses[i]

        cls.fit(X, np.ravel(y))
        cls_vae.fit(X_vae, np.ravel(y_vae))
        cls_smote.fit(X_smote, np.ravel(y_smote))

        display = PrecisionRecallDisplay.from_estimator(cls, X_valid, np.ravel(y_valid), ax=plt.gca(), name="original")
        display = PrecisionRecallDisplay.from_estimator(cls_vae, X_valid, np.ravel(y_valid), ax=plt.gca(), name='vae')
        display = PrecisionRecallDisplay.from_estimator(cls_smote, X_valid, np.ravel(y_valid), ax=plt.gca(), name='smote')
        plt.savefig(save_loc+cls.__class__.__name__+'_PRC.pdf')
        plt.show()



dataset_attributes = {
    'breast_cancer_coimbra':{
        'cat_idxs':[],
        'int_idxs':[0,2],
        'float_idxs':[1,3,4,5,6,7,8],
        'generate_num':100,
    },
    'diabetes_risk_prediction':{
        'cat_idxs':[],
        'int_idxs':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        'float_idxs':[],
        'generate_num':400,
    },
    'framingham':{
        'cat_idxs':[2],
        'int_idxs':[0,1,3,4,5,6,7,8,9,13,14],
        'float_idxs':[10,11,12],
        'generate_num':2000,
    },
    'hepatitis':{
        'cat_idxs':[],
        'int_idxs':[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,17,18],
        'float_idxs':[13,16],
        'generate_num':100,
    },
    'kidney_disease':{
        'cat_idxs':[3,4],
        'int_idxs':[0,1,5,6,7,8,9,15,16,18,19,20,21,22,23],
        'float_idxs':[2,10,11,12,13,14,17],
        'generate_num':250,
    },
    'PIMA_diabetes':{
        'cat_idxs':[],
        'int_idxs':[0,1,2,3,4,7],
        'float_idxs':[5,6],
        'generate_num':300,
    },
    'TCGA_InfoWithGrade':{
        'cat_idxs':[2],
        'int_idxs':[0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],
        'float_idxs':[1],
        'generate_num':200,
    },
    'Thyroid_Diff':{
        'cat_idxs':[5,6,7,8,10,11,12,13,14,15],
        'int_idxs':[0,1,2,3,4,9],
        'float_idxs':[],
        'generate_num':200,
    },
    'yc_diabetes':{
        'cat_idxs':[25,26,27,28,29,32,33,35,36,37,38,39,40,41,42,43,44],
        'int_idxs':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,30,31,34],
        'float_idxs':[],
        'generate_num':13000,
    },
    'yc_hyper':{
        'cat_idxs':[25,26,27,28,29,32,33,35,36,37,38,39,40,41,42,43,44],
        'int_idxs':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,30,31,34],
        'float_idxs':[],
        'generate_num':10000,
    }
}


#=================================================================================================================
#=================================================================================================================

from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, f1_score
from imblearn.metrics import geometric_mean_score

def standard_classification_test(X_train, y_train, X_test, y_test, ir, seed=42, meta=None):
    
    n_samples = len(X_train)
    meta = meta or {}
    
    # 1. 初始化具有自适应参数的分类器集群
    clfs = {
        'SVC': LinearSVC(dual=False, class_weight='balanced', C=0.05, 
                         max_iter=5000, random_state=seed, tol=1e-4),
        
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), 
                             max_iter=1000, 
                             alpha=0.01 if n_samples > 500 else 0.1, 
                             early_stopping=True if n_samples > 200 else False,
                             validation_fraction=0.1 if n_samples > 500 else 0.2,
                             n_iter_no_change=20,
                             random_state=seed),
        
        'XGB': XGBClassifier(n_estimators=200 if n_samples > 500 else 100, 
                             max_depth=5 if n_samples > 500 else 3, 
                             learning_rate=0.05,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             scale_pos_weight=ir,
                             eval_metric='logloss',
                             random_state=42) # XGB 随机性较小，固定种子
    }
    
    results = []
    
   
    for name, clf in clfs.items():
       
        clf.fit(X_train.values if hasattr(X_train, 'values') else X_train, 
                y_train.values.ravel() if hasattr(y_train, 'values') else y_train.ravel())
        
        
        y_pred = clf.predict(X_test.values if hasattr(X_test, 'values') else X_test)
        y_true = y_test.values.ravel() if hasattr(y_test, 'values') else y_test.ravel()
        
        
        base_row = {**meta, 'Classifier': name}
        
        results.append({**base_row, 'Metric': 'Recall', 'Value': recall_score(y_true, y_pred)})
        results.append({**base_row, 'Metric': 'F1_score', 'Value': f1_score(y_true, y_pred)})
        results.append({**base_row, 'Metric': 'G_mean', 'Value': geometric_mean_score(y_true, y_pred)})
        
    return results