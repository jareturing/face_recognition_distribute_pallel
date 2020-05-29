# -*- coding: utf-8 -*-
###############################################################################
# 查询接口的IO定义,rewritten by numpy
# @author: Zihan Meng
# @date: 2019/08/08
# @edit:2020.04.21
###############################################################################
import os
import numpy as np

class SearchResult():
    """
    1：N查询结果结构定义
    QueryID:QueryInput
    SearchID：查询结果id，int
    SearchResultID:降序排列 list
    SearchResultSimilarity：降序排列，np.array
    """

    def __init__(self):
        self.SearchID = None
        self.QueryID = None
        self.SearchResultID = None
        self.SearchResultSimilarity = None


class SearchResultCompare():
    """
    1比1查询结果结构定义
    QueryID:查询id str
    SearchID：查询结果id int
    SearchResultID:查中id str
    SearchResultSimilarity：相似度结果，float
    Type:positive or negative

    """

    def __init__(self, SearchID, QueryID, SearchResultID, SearchResultSimilarity, Type):
        self.SearchID = SearchID
        self.QueryID = QueryID
        self.SearchResultID = SearchResultID
        self.SearchResultSimilarity = SearchResultSimilarity
        self.TestType = Type


class QueryInput():
    """
    1:n查询数据结构定义
    """

    def __init__(self, queryname, flag):
        self.queryname = queryname
        self.flag = flag


class Search():
    """
    1:N计算搜索结果
    """

    def __init__(self):
        self.all_search_reuslt = []
        self.flag = 0
        self.huoti_num = 0
        self.zhengjian_num = 0
        self.topN = 50

    def search(self, Queryname, QueryFeature, Galleryname, GalleryFeature):
        for id, queryname in enumerate(Queryname):
            if queryname.split('/')[-2] == 'zhengjian':
                flag = 1
                self.zhengjian_num += 1
            else:
                flag = 2
                self.huoti_num += 1
            # print(self.zhengjian_num, self.huoti_num)
            img = QueryInput(os.path.basename(queryname).replace(".jpg", ''), flag)
            result = SearchResult()
            result.SearchID = id
            result.QueryID = img
            result.SearchResultID = []
            query_result = np.dot(QueryFeature[id], GalleryFeature.T)
            sim_result_index = np.argsort(-query_result)[:self.topN].astype(np.int32)
            result.SearchResultSimilarity = query_result[sim_result_index]
            for i in range(sim_result_index.shape[0]):
                index = sim_result_index[i]
                # print(index)
                galleryname = os.path.basename(Galleryname[int(index)]).replace(".jpg", '')
                result.SearchResultID.append(galleryname)
            self.all_search_reuslt.append(result)
            if len(self.all_search_reuslt) % 100 == 0:
                print("finished {}".format(len(self.all_search_reuslt)))
        return self.all_search_reuslt

class Verify():
    """
    1:1验证结果
    """
    def __init__(self, paristxt, datadir):
        self.pairstxit = paristxt
        self.datadir = datadir

    def verify(self, Feature, query_imgs, pairs_list):
        """
        1:1验证
        :param Feature: 人脸特征
        :return:
        """
        all_search_reuslt = []
        for id, pair in enumerate(pairs_list):
            queryname = os.path.basename(query_imgs[id * 2])
            searchname = os.path.basename(query_imgs[id * 2 + 1])
            simlarity = np.dot(Feature[id * 2], Feature[id * 2 + 1].T)
            result = SearchResultCompare(SearchID=id,
                                         QueryID=queryname,
                                         SearchResultID=searchname,
                                         SearchResultSimilarity=simlarity,
                                         Type=pair)
            all_search_reuslt.append(result)
        return all_search_reuslt


