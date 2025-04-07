import pandas as pd
import numpy as np
def split(x_series,y,min_samples_split):
    import pandas as pd
    import numpy as np
    import math
    dataframe_tmp = pd.DataFrame({'x': x_series, 'y': y}).sort_values(by='x', ascending=True)
    dataframe_tmp=dataframe_tmp.reset_index(inplace=False)
    split_point_score = {}
    gini_group={}
    if len(x_series)<min_samples_split:
        print('样本量不足无法分裂，计算中止')
        return np.NaN,np.NaN,np.NaN,[np.NaN,np.NaN,np.NaN]
    else:
        for i in range(0,len(x_series)):
            if i<min_samples_split:
                pass
            else:
                for j in range(i,len(x_series)):
                    if len(x_series)-j<min_samples_split:
                        pass
                    elif j - i < min_samples_split:
                        pass
                    elif i==j:
                        gini1=1-pow(dataframe_tmp[:i+1].groupby('y').count()/(i+1),2).sum()[0]
                        gini2=np.NaN
                        gini3=1-pow(dataframe_tmp[i+1:].groupby('y').count()/(len(x_series)-i-1),2).sum()[0]
                        split_point_score[str(i)+'_'+str(j)]=(i+1)/len(x_series)*gini1+(len(x_series)-i-1)/len(x_series)*gini3
                        gini_group[str(dataframe_tmp.loc[i,'x'])+'_'+str(dataframe_tmp.loc[j,'x'])]=[gini1,gini2,gini3]
                    else:
                        gini1 = 1 -pow(dataframe_tmp[:i+1].groupby('y').count() / (i + 1), 2).sum()[0]
                        gini2 = 1 -pow(dataframe_tmp[i+1:j+1].groupby('y').count() / (j-i), 2).sum()[0]
                        gini3 = 1 - pow(dataframe_tmp[j+1:].groupby('y').count() / (len(x_series) - j - 1),
                                2).sum()[0]
                        split_point_score[str(dataframe_tmp.loc[i,'x'])+'_'+str(dataframe_tmp.loc[j,'x'])] = (i + 1) / len(x_series) * gini1 + \
                                                                   (j - i)/len(x_series) * gini2 + \
                                                                   (len(x_series) - j - 1) / len(x_series) * gini3
                        gini_group[str(dataframe_tmp.loc[i,'x'])+'_'+str(dataframe_tmp.loc[j,'x'])] = [gini1, gini2, gini3]
        try:
            split_point = min(split_point_score.keys(),key=(lambda x: split_point_score[x]))
            gini=split_point_score[split_point]
            [gini1,gini2,gini3]=gini_group[split_point]
            return split_point.split('_')[0],split_point.split('_')[1],gini,[gini1,gini2,gini3]
        except:
            print('样本小于min_samples_split')
            return np.NaN,np.NaN,np.NaN,[np.NaN,np.NaN,np.NaN]
def choose_best_feature(x, y, min_samples_split):
    gini_list = {}
    split_1_list={}
    split_2_list = {}
    gini_group_list = {}
    for x_col in x.columns:
        split_1, split_2, gini,[gini1,gini2,gini3] = split(x[x_col], y, min_samples_split)
        if split_1!=np.NaN:
            gini_list[x_col] = gini
            split_1_list[x_col] = split_1
            split_2_list[x_col] = split_2
            gini_group_list[x_col] = [gini1,gini2,gini3]
    if len(gini_list.keys())>0:
        best_feature = min(gini_list.keys(), key=(lambda x: gini_list[x]))
        return best_feature,split_1_list[best_feature],\
           split_2_list[best_feature],\
           gini_list[best_feature],\
           gini_group_list[best_feature]
    else:
        np.NaN,np.NaN,np.NaN,np.NaN,[np.NaN, np.NaN, np.NaN]


def create_decision_tree_first_split(x, y, min_samples_split, min_gini, min_samples_leaf, num):
    import pandas as pd
    import numpy as np
    import math
    best_feature, best_split_1, best_split_2, gini, [gini1, gini2, gini3] \
        = choose_best_feature(x, y, min_samples_split)
    tree = {best_feature: {}}
    if str(best_split_1) == 'nan':
        print('样本分裂后的子样本小于min_samples_split')
        dataframe_tmp = pd.DataFrame(x)
        dataframe_tmp['y'] = y
        tree = dataframe_tmp.groupby('y')[best_feature].count() \
            [dataframe_tmp.groupby('y')[best_feature].count() ==
             max(dataframe_tmp.groupby('y')[best_feature].count())].index[0]
        return tree
    # elif gini <= 0.05:
    #     print('样本GINI指数太小，没意义分裂')
    #     dataframe_tmp = pd.DataFrame(x)
    #     dataframe_tmp['y'] = y
    #     tree = dataframe_tmp.groupby('y')[best_feature].count() \
    #         [dataframe_tmp.groupby('y')[best_feature].count() ==
    #          max(dataframe_tmp.groupby('y')[best_feature].count())].index[0]
    #     return tree
    else:
        best_split_1 = float(best_split_1)
        best_split_2 = float(best_split_2)
        print('分裂的最优列:', best_feature)
        print(best_split_1, best_split_2)
        dataframe_tmp = pd.DataFrame(x)
        dataframe_tmp['y'] = y
        subtree = {}
        if str(gini2) == 'nan':
            if gini1 < min_gini or num == 3 or len(
                    dataframe_tmp[dataframe_tmp[best_feature] <= best_split_1]) <= 2 * min_samples_leaf:
                y1 = dataframe_tmp[dataframe_tmp[best_feature] <= best_split_1].groupby('y')[best_feature].count() \
                    [
                    dataframe_tmp[dataframe_tmp[best_feature] <= best_split_1].groupby('y')[best_feature].count() ==
                    max(dataframe_tmp[dataframe_tmp[best_feature] <= best_split_1].groupby('y')[
                            best_feature].count())].index[0]
                subtree['<=' + str(best_split_1)] = y1
            else:
                subtree['<=' + str(best_split_1)] = {}
            if gini3 < min_gini or num == 3 or len(
                    dataframe_tmp[dataframe_tmp[best_feature] > best_split_2]) <= 2 * min_samples_leaf:
                y2 = dataframe_tmp[dataframe_tmp[best_feature] > best_split_2].groupby('y')[best_feature].count() \
                    [dataframe_tmp[dataframe_tmp[best_feature] > best_split_2].groupby('y')[best_feature].count() ==
                     max(dataframe_tmp[dataframe_tmp[best_feature] > best_split_2].groupby('y')[
                             best_feature].count())].index[0]
                subtree['>=' + str(best_split_2)] = y2
            else:
                subtree['>=' + str(best_split_2)] = {}
        else:
            if gini1 < min_gini or num == 3 or len(
                    dataframe_tmp[dataframe_tmp[best_feature] <= best_split_1]) <= 2 * min_samples_leaf:
                print('123456: ', dataframe_tmp[dataframe_tmp[best_feature] <= best_split_1])
                y1 = dataframe_tmp[dataframe_tmp[best_feature] <= best_split_1].groupby('y')[best_feature].count() \
                    [
                    dataframe_tmp[dataframe_tmp[best_feature] <= best_split_1].groupby('y')[best_feature].count() ==
                    max(dataframe_tmp[dataframe_tmp[best_feature] <= best_split_1].groupby('y')[
                            best_feature].count())].index[0]
                subtree['<=' + str(best_split_1)] = y1
            else:
                subtree['<=' + str(best_split_1)] = {}
            if gini2 < min_gini or num == 3 or len(dataframe_tmp[(dataframe_tmp[best_feature] > best_split_1) & (
                    dataframe_tmp[best_feature] <= best_split_2)]) <= 2 * min_samples_leaf:
                print(dataframe_tmp[(dataframe_tmp[best_feature] > best_split_1) & (
                        dataframe_tmp[best_feature] <= best_split_2)].groupby('y')[best_feature].count())
                y2 = dataframe_tmp[(dataframe_tmp[best_feature] > best_split_1) & (
                        dataframe_tmp[best_feature] <= best_split_2)].groupby('y')[best_feature].count() \
                    [dataframe_tmp[(dataframe_tmp[best_feature] > best_split_1) & (
                            dataframe_tmp[best_feature] <= best_split_2)].groupby('y')[best_feature].count() ==
                     max(dataframe_tmp[(dataframe_tmp[best_feature] > best_split_1) & (
                             dataframe_tmp[best_feature] <= best_split_2)].groupby('y')[
                             best_feature].count())].index[0]
                subtree[str(best_split_1) + '-' + str(best_split_2)] = y2
            else:
                subtree[str(best_split_1) + '-' + str(best_split_2)] = {}
            if gini3 < min_gini or num == 3 or len(
                    dataframe_tmp[dataframe_tmp[best_feature] > best_split_1]) <= 2 * min_samples_leaf:
                y2 = dataframe_tmp[dataframe_tmp[best_feature] > best_split_1].groupby('y')[best_feature].count() \
                    [dataframe_tmp[dataframe_tmp[best_feature] > best_split_1].groupby('y')[best_feature].count() ==
                     max(dataframe_tmp[dataframe_tmp[best_feature] > best_split_1].groupby('y')[
                             best_feature].count())].index[0]
                subtree['>=' + str(best_split_2)] = y2
            else:
                subtree['>=' + str(best_split_2)] = {}
        tree[best_feature] = subtree
        return tree


def create_decision_tree_split_twice(x, y, min_samples_split, min_gini, min_samples_leaf, tree):
    import pandas as pd
    import numpy as np
    print(x)
    dataframe_tmp = pd.DataFrame(x)
    dataframe_tmp['y'] = y
    # 第二次分裂
    for key, value in tree.items():
        for key_1, value_1 in value.items():
            if value_1 == {}:
                if '<=' in key_1:
                    x1 = dataframe_tmp[dataframe_tmp[key] <= float(key_1.replace('<=', ''))].drop(columns=['y'])
                    y1 = dataframe_tmp[dataframe_tmp[key] <= float(key_1.replace('<=', ''))]['y']
                    subtree = create_decision_tree_first_split(x1, y1, min_samples_split, min_gini,
                                                                    min_samples_leaf, num=2)
                    tree[key][key_1] = subtree
                elif '-' in key_1:
                    x1 = dataframe_tmp[(dataframe_tmp[key] <= float(key_1.split('-')[1])) & (
                            dataframe_tmp[key] > float(key_1.split('-')[0]))].drop(columns=['y'])
                    y1 = dataframe_tmp[(dataframe_tmp[key] <= float(key_1.split('-')[1])) & (
                            dataframe_tmp[key] > float(key_1.split('-')[0]))]['y']
                    subtree = create_decision_tree_first_split(x1, y1, min_samples_split, min_gini,
                                                                    min_samples_leaf, num=2)
                    tree[key][key_1] = subtree
                elif '>=' in key_1:
                    x1 = dataframe_tmp[dataframe_tmp[key] > float(key_1.replace('>=', ''))].drop(columns=['y'])
                    y1 = dataframe_tmp[dataframe_tmp[key] > float(key_1.replace('>=', ''))]['y']
                    subtree = create_decision_tree_first_split(x1, y1, min_samples_split, min_gini,
                                                                    min_samples_leaf, num=2)
                    tree[key][key_1] = subtree
        # 第三次分裂
        for key_1, value_1 in value.items():
            if type(value_1) == dict and value_1 != {}:
                for key_2, value_2 in value_1.items():
                    for key_3, value_3 in value_2.items():
                        if value_3 == {}:
                            if '<=' in key_3:
                                if '<=' in key_1:
                                    x1 = dataframe_tmp[dataframe_tmp[key] <= float(key_1.replace('<=', ''))][
                                        dataframe_tmp[key_2] <= float(key_3.replace('<=', ''))].drop(columns=['y'])
                                    y1 = dataframe_tmp[dataframe_tmp[key] <= float(key_1.replace('<=', ''))][
                                        dataframe_tmp[key_2] <= float(key_3.replace('<=', ''))]['y']
                                elif '-' in key_1:
                                    x1 = dataframe_tmp[(dataframe_tmp[key] <= float(key_1.split('-')[1])) & (
                                            dataframe_tmp[key] > float(key_1.split('-')[0]))][
                                        dataframe_tmp[key_2] <= float(key_3.replace('<=', ''))].drop(columns=['y'])
                                    y1 = dataframe_tmp[(dataframe_tmp[key] <= float(key_1.split('-')[1])) & (
                                            dataframe_tmp[key] > float(key_1.split('-')[0]))][
                                        dataframe_tmp[key_2] <= float(key_3.replace('<=', ''))]['y']
                                elif '>=' in key_1:
                                    x1 = dataframe_tmp[dataframe_tmp[key] > float(key_1.replace('>=', ''))][
                                        dataframe_tmp[key_2] <= float(key_3.replace('<=', ''))].drop(columns=['y'])
                                    y1 = dataframe_tmp[dataframe_tmp[key] > float(key_1.replace('>=', ''))][
                                        dataframe_tmp[key_2] <= float(key_3.replace('<=', ''))]['y']
                                subtree = create_decision_tree_first_split(x1, y1, min_samples_split, min_gini,
                                                                                min_samples_leaf, num=3)

                                if type(subtree) == dict:
                                    for min_key, min_value in subtree.items():
                                        if subtree[min_key] == {}:
                                            subtree[min_key] = 0
                                tree[key][key_1][key_2][key_3] = subtree
                            elif '-' in key_3:
                                if '<=' in key_1:
                                    x1 = dataframe_tmp[dataframe_tmp[key] <= float(key_1.replace('<=', ''))][
                                        (dataframe_tmp[key_2] <= float(key_3.split('-')[1])) & (
                                                dataframe_tmp[key_2] > float(key_3.split('-')[0]))].drop(
                                        columns=['y'])
                                    y1 = dataframe_tmp[dataframe_tmp[key] <= float(key_1.replace('<=', ''))][
                                        (dataframe_tmp[key_2] <= float(key_3.split('-')[1])) & (
                                                dataframe_tmp[key_2] > float(key_3.split('-')[0]))]['y']
                                elif '-' in key_1:
                                    x1 = dataframe_tmp[(dataframe_tmp[key] <= float(key_1.split('-')[1])) & (
                                            dataframe_tmp[key] > float(key_1.split('-')[0]))][
                                        (dataframe_tmp[key_2] <= float(key_3.split('-')[1])) & (
                                                dataframe_tmp[key_2] > float(key_3.split('-')[0]))].drop(
                                        columns=['y'])
                                    y1 = dataframe_tmp[(dataframe_tmp[key] <= float(key_1.split('-')[1])) & (
                                            dataframe_tmp[key] > float(key_1.split('-')[0]))][
                                        (dataframe_tmp[key_2] <= float(key_3.split('-')[1])) & (
                                                dataframe_tmp[key_2] > float(key_3.split('-')[0]))]['y']
                                elif '>=' in key_1:
                                    x1 = dataframe_tmp[dataframe_tmp[key] > float(key_1.replace('>=', ''))][
                                        (dataframe_tmp[key_2] <= float(key_3.split('-')[1])) & (
                                                dataframe_tmp[key_2] > float(key_3.split('-')[0]))].drop(
                                        columns=['y'])
                                    y1 = dataframe_tmp[dataframe_tmp[key] > float(key_1.replace('>=', ''))][
                                        (dataframe_tmp[key_2] <= float(key_3.split('-')[1])) & (
                                                dataframe_tmp[key_2] > float(key_3.split('-')[0]))]['y']
                                subtree = create_decision_tree_first_split(x1, y1, min_samples_split, min_gini,
                                                                                min_samples_leaf, num=3)
                                if type(subtree) == dict:
                                    for min_key, min_value in subtree.items():
                                        if subtree[min_key] == {}:
                                            subtree[min_key] = 0
                                tree[key][key_1][key_2][key_3] = subtree
                            elif '>=' in key_3:
                                if '<=' in key_1:
                                    x1 = dataframe_tmp[dataframe_tmp[key] <= float(key_1.replace('<=', ''))][
                                        dataframe_tmp[key_2] > float(key_3.replace('>=', ''))].drop(columns=['y'])
                                    y1 = dataframe_tmp[dataframe_tmp[key] <= float(key_1.replace('<=', ''))][
                                        dataframe_tmp[key_2] > float(key_3.replace('>=', ''))]['y']
                                elif '-' in key_1:
                                    x1 = dataframe_tmp[(dataframe_tmp[key] <= float(key_1.split('-')[1])) & (
                                            dataframe_tmp[key] > float(key_1.split('-')[0]))][
                                        dataframe_tmp[key_2] > float(key_3.replace('>=', ''))].drop(columns=['y'])
                                    y1 = dataframe_tmp[(dataframe_tmp[key] <= float(key_1.split('-')[1])) & (
                                            dataframe_tmp[key] > float(key_1.split('-')[0]))][
                                        dataframe_tmp[key_2] > float(key_3.replace('>=', ''))]['y']
                                elif '>=' in key_1:
                                    x1 = dataframe_tmp[dataframe_tmp[key] > float(key_1.replace('>=', ''))][
                                        dataframe_tmp[key_2] > float(key_3.replace('>=', ''))].drop(columns=['y'])
                                    y1 = dataframe_tmp[dataframe_tmp[key] > float(key_1.replace('>=', ''))][
                                        dataframe_tmp[key_2] > float(key_3.replace('>=', ''))]['y']
                                subtree = create_decision_tree_first_split(x1, y1, min_samples_split, min_gini,
                                                                                min_samples_leaf, num=3)
                                if type(subtree) == dict:
                                    for min_key, min_value in subtree.items():
                                        if subtree[min_key] == {}:
                                            subtree[min_key] = 0
                                tree[key][key_1][key_2][key_3] = subtree
    print(tree)
    return tree
def predict(x_predict,tree):
    y=0
    for key_1,value_1 in tree.items():
        for key_2,value_2 in value_1.items():
            if '<=' in key_2 and x_predict[key_1].item() <= float(key_2.replace('<=', '')):
                if value_2 in [0,1]:
                    y=value_2
                    print(1)
                    break
                else:
                    for key_3,value_3 in value_2.items():
                        for key_4, value_4 in value_3.items():
                            if '<=' in key_4 and x_predict[key_3].item() <= float(key_4.replace('<=', '')):
                                if value_4 in [0, 1]:
                                    y = value_4
                                    print(2)
                                    break
                                else:
                                    for key_5, value_5 in value_4.items():
                                        for key_6, value_6 in value_5.items():
                                            if '<=' in key_6 and x_predict[key_5].item() <= float(key_6.replace('<=', '')):
                                                y = value_6
                                                print(3)
                                                break
                                            elif '-' in key_6 and x_predict[key_5].item() > float(key_6.split('-')[0]) and x_predict[key_5].item() <= float(key_6.split('-')[1]):
                                                y = value_6
                                                print(4)
                                                break
                                            elif '>=' in key_6 and x_predict[key_5].item() <= float(key_6.replace('>=', '')):
                                                y = value_6
                                                print(5)
                                                break
                                            else:
                                                pass
                            elif '-' in key_4 and x_predict[key_3].item() > float(key_4.split('-')[0]) and x_predict[key_3].item() <= float(key_4.split('-')[1]):
                                if value_4 in [0, 1]:
                                    y = value_4
                                    print(6)
                                    break
                                else:
                                    for key_5, value_5 in value_4.items():
                                        for key_6, value_6 in value_5.items():
                                            if '<=' in key_6 and x_predict[key_5].item() <= float(key_6.replace('<=', '')):
                                                y = value_6
                                                print(7)
                                                break
                                            elif '-' in key_6 and x_predict[key_5].item() > float(key_6.split('-')[0]) and x_predict[key_5].item() <= float(key_6.split('-')[1]):
                                                y = value_6
                                                print(8)
                                                break
                                            elif '>=' in key_6 and x_predict[key_5].item() <= float(key_6.replace('>=', '')):
                                                y = value_6
                                                print(9)
                                                break
                                            else:
                                                pass
                            elif '>=' in key_4 and x_predict[key_3].item() > float(key_4.replace('>=', '')):
                                if value_4 in [0, 1]:
                                    y = value_4
                                    print(10)
                                    break
                                else:
                                    for key_5, value_5 in value_4.items():
                                        for key_6, value_6 in value_5.items():
                                            if '<=' in key_6 and x_predict[key_5].item() <= float(key_6.replace('<=', '')):
                                                y = value_6
                                                print(11)
                                                break
                                            elif '-' in key_6 and x_predict[key_5].item() > float(key_6.split('-')[0]) and x_predict[key_5].item() <= float(key_6.split('-')[1]):
                                                y = value_6
                                                print(12)
                                                break
                                            elif '>=' in key_6 and x_predict[key_5].item() <= float(
                                                    key_6.replace('>=', '')):
                                                y = value_6
                                                print(13)
                                                break
                                            else:
                                                pass
                            else:
                                pass
            elif '-' in key_2 and x_predict[key_1].item() > float(key_2.split('-')[0]) and x_predict[key_1].item() <= float(
                key_2.split('-')[1]):
                if value_2 in [0,1]:
                    y=value_2
                    print(14)
                    break
                else:
                    for key_3,value_3 in value_2.items():
                        for key_4, value_4 in value_3.items():
                            if '<=' in key_4 and x_predict[key_3].item() <= float(key_4.replace('<=', '')):
                                if value_4 in [0, 1]:
                                    y = value_4
                                    print(15)
                                    break
                                else:
                                    for key_5, value_5 in value_4.items():
                                        for key_6, value_6 in value_5.items():
                                            if '<=' in key_6 and x_predict[key_5].item() <= float(key_6.replace('<=', '')):
                                                y = value_6
                                                print(16)
                                                break
                                            elif '-' in key_6 and x_predict[key_5].item() > float(key_6.split('-')[0]) and x_predict[key_5].item() <= float(key_6.split('-')[1]):
                                                y = value_6
                                                print(17)
                                                break
                                            elif '>=' in key_6 and x_predict[key_5].item() <= float(
                                                    key_6.replace('>=', '')):
                                                y = value_6
                                                print(18)
                                                break
                                            else:
                                                pass
                            elif '-' in key_4 and x_predict[key_3].item() > float(key_4.split('-')[0]) and x_predict[key_3].item() <= float(key_4.split('-')[1]):
                                if value_4 in [0, 1]:
                                    y = value_4
                                else:
                                    for key_5, value_5 in value_4.items():
                                        for key_6, value_6 in value_5.items():
                                            if '<=' in key_6 and x_predict[key_5].item() <= float(key_6.replace('<=', '')):
                                                y = value_6
                                                print(19)
                                                break
                                            elif '-' in key_6 and x_predict[key_5].item() > float(key_6.split('-')[0]) and x_predict[key_5].item() <= float(key_6.split('-')[1]):
                                                y = value_6
                                                print(20)
                                                break
                                            elif '>=' in key_6 and x_predict[key_5].item() <= float(
                                                    key_6.replace('>=', '')):
                                                y = value_6
                                                print(21)
                                                break
                                            else:
                                                pass
                            elif '>=' in key_4 and x_predict[key_3].item() > float(key_4.replace('>=', '')):
                                if value_4 in [0, 1]:
                                    y = value_4
                                    print(22)
                                    break
                                else:
                                    for key_5, value_5 in value_4.items():
                                        for key_6, value_6 in value_5.items():
                                            if '<=' in key_6 and x_predict[key_5].item() <= float(key_6.replace('<=', '')):
                                                y = value_6
                                                print(23)
                                                break
                                            elif '-' in key_6 and x_predict[key_5].item() > float(key_6.split('-')[0]) and x_predict[key_5].item() <= float(key_6.split('-')[1]):
                                                y = value_6
                                                print(24)
                                                break
                                            elif '>=' in key_6 and x_predict[key_5].item() <= float(
                                                    key_6.replace('>=', '')):
                                                y = value_6
                                                print(25)
                                                break
                                            else:
                                                pass
                            else:
                                pass
            elif '>=' in key_2 and x_predict[key_1].item() > float(key_2.replace('>=', '')):
                if value_2 in [0,1]:
                    y=value_2
                    print(26)
                    break
                else:
                    for key_3,value_3 in value_2.items():
                        for key_4, value_4 in value_3.items():
                            if '<=' in key_4 and x_predict[key_3].item() <= float(key_4.replace('<=', '')):
                                if value_4 in [0, 1]:
                                    y = value_4
                                    print(27)
                                    break
                                else:
                                    for key_5, value_5 in value_4.items():
                                        for key_6, value_6 in value_5.items():
                                            if '<=' in key_6 and x_predict[key_5].item() <= float(key_6.replace('<=', '')):
                                                y = value_6
                                                print(28)
                                                break
                                            elif '-' in key_6 and x_predict[key_5].item() > float(key_6.split('-')[0]) and x_predict[key_5].item() <= float(key_6.split('-')[1]):
                                                y = value_6
                                                print(29)
                                                break
                                            elif '>=' in key_6 and x_predict[key_5].item() <= float(
                                                    key_6.replace('>=', '')):
                                                y = value_6
                                                print(30)
                                                break
                                            else:
                                                pass
                            elif '-' in key_4 and x_predict[key_3].item() > float(key_4.split('-')[0]) and x_predict[key_3].item() <= float(key_4.split('-')[1]):
                                if value_4 in [0, 1]:
                                    y = value_4
                                    print(31)
                                    break
                                else:
                                    for key_5, value_5 in value_4.items():
                                        for key_6, value_6 in value_5.items():
                                            if '<=' in key_6 and x_predict[key_5].item() <= float(key_6.replace('<=', '')):
                                                y = value_6
                                                print(32)
                                                break
                                            elif '-' in key_6 and x_predict[key_5].item() > float(key_6.split('-')[0]) and x_predict[key_5].item() <= float(key_6.split('-')[1]):
                                                y = value_6
                                                print(33)
                                                break
                                            elif '>=' in key_6 and x_predict[key_5].item() <= float(
                                                    key_6.replace('>=', '')):
                                                y = value_6
                                                print(34)
                                                break
                                            else:
                                                pass
                            elif '>=' in key_4 and x_predict[key_3].item() > float(key_4.replace('>=', '')):
                                if value_4 in [0, 1]:
                                    y = value_4
                                    print(35)
                                    break
                                else:
                                    for key_5, value_5 in value_4.items():
                                        for key_6, value_6 in value_5.items():
                                            if '<=' in key_6 and x_predict[key_5].item() <= float(key_6.replace('<=', '')):
                                                y = value_6
                                                print(36)
                                                break
                                            elif '-' in key_6 and x_predict[key_5].item() > float(key_6.split('-')[0]) and x_predict[key_5].item() <= float(key_6.split('-')[1]):
                                                y = value_6
                                                print(37)
                                                break
                                            elif '>=' in key_6 and x_predict[key_5].item() <= float(
                                                    key_6.replace('>=', '')):
                                                y = value_6
                                                print(38)
                                                break
                                            else:
                                                pass
                            else:
                                pass
            else:
                pass
    return y
def plot_auc(y_predict,y_test):
    from sklearn import metrics
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_predict)
    print(fpr, tpr, threshold)
    rocauc = metrics.auc(fpr, tpr)
    print(rocauc)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))  # 只能在这里面设置
    plt.plot(fpr, tpr, 'b', label='AUC=%0.2f' % rocauc)
    plt.legend(loc='lower right', fontsize=14)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # 增加坐标轴刻度#
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('TPR', fontsize=16)
    plt.xlabel('FPR', fontsize=16)
    plt.title('ROC CURVE', fontsize=12, color='red')
    plt.show()
def confussion_matrix(y_predict,y_test):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_test,y_predict)
def classification_report(y_predict,y_test):
    from sklearn.metrics import classification_report
    report=classification_report(y_test,y_predict,output_dict=True)
    return report
# def predict(self,x_predict,tree):
data=pd.read_csv('C:/Users/任英睿/PycharmProjects/pythonProject/Financial_risk/TrainData.csv',encoding='gbk')\
[20:100]
x=data[['RevolvingUtilizationOfUnsecuredLines','age','DebtRatio','MonthlyIncome']]
y=data['SeriousDlqin2yrs']
tree=create_decision_tree_first_split(x, y, 2, 0.05, 2,num=1)
tree=create_decision_tree_split_twice(x, y, 2, 0.05, 2, tree)
x_predict=pd.read_csv('C:/Users/任英睿/PycharmProjects/pythonProject/Financial_risk/TrainData.csv',encoding='gbk')[2000:2100][['RevolvingUtilizationOfUnsecuredLines','age','DebtRatio','MonthlyIncome']]
y_predict=x_predict.apply(lambda x:predict(x,tree),axis=1)
y_true=pd.read_csv('C:/Users/任英睿/PycharmProjects/pythonProject/Financial_risk/TrainData.csv',encoding='gbk')[2000:2100]['SeriousDlqin2yrs']
plot_auc(y_predict,y_true)
data_confusion=confussion_matrix(y_predict,y_true)
pd.DataFrame(data_confusion).to_excel('C:/Users/任英睿/PycharmProjects/pythonProject/cart树改进/data_confusion_3.xlsx',index=False)
report=classification_report(y_predict,y_true)
pd.DataFrame(report).to_excel('C:/Users/任英睿/PycharmProjects/pythonProject/cart树改进/report_3.xlsx',index=False)