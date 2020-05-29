# -*- coding: utf-8 -*-
###############################################################################
# 评估类的指标计算方法
# @author: Zihan Meng
# @date: 2019/08/08
###############################################################################
from io_defination import *
import numpy as np


class Metrics():
    def __init__(self, SearchResutlist):
        self.SearchResutlist = SearchResutlist

    def get_first_match(self, querynum, flag):  # top-1 recall
        """
        计算首位匹配率
        :param querynum:探测集样本总数 int
        :param flag: 是证件照还是活体照标志 int
        :return: 首位匹配率 float
        """
        count = 0
        for searchresult in self.SearchResutlist:
            queryid = searchresult.QueryID
            # print(queryid.queryname.split('_')[0] , searchresult.SearchResultID[0])
            if queryid.flag == flag and queryid.queryname.split('_')[0] == searchresult.SearchResultID[0]:
                count += 1

        return round(float(count) / float(querynum), 4)

    def get_topn_match(self, querynum, flag, topn):  # top-N recall
        """
        计算前n位匹配率
        :param querynum: 探测集样本总数 int
        :param flag: 是证件照还是活体照标志 int
        :param topn: 前n位
        :return: 前n位匹配率 float
        """
        count = 0
        for searchresult in self.SearchResutlist:
            queryid = searchresult.QueryID
            if queryid.flag == flag:
                for i in range(topn):
                    if queryid.queryname.split('_')[0] == searchresult.SearchResultID[i]:
                        count += 1
                        break
        return round(float(count) / float(querynum), 4)

    def get_target_match(self, querynum):  # recall，以quary_id in search_result为tp
        """
        计算查中率
        :param querynum: 探测集样本总数 int
        :return:查中率 float
        """
        count = 0
        for searchresult in self.SearchResutlist:
            queryid = searchresult.QueryID
            if queryid.queryname.split('_')[0] in searchresult.SearchResultID:
                count += 1
        return round(float(count) / float(querynum), 4)

    def get_fppiundertarget_match(self, targetrate, querynum):  # 确定阈值（tp 最小的相似度），计算fp rate FP/(TP+FP)
        """
        计算确定查中率下的误报率
        :param targetrate: 查中率 float
        :param querynum: 探测集样本总数 int
        :return: 确定查中率下的误报率
        """
        # 前n位累积匹配结果中相似度大于等于阈值T的所有结果数
        more_threshold_result_num = 0
        # 前n位累积匹配结果中相似度大于等于阈值T的所有误识结果数
        more_threshold_falseresult_num = 0
        # 得到查中数
        targetnum = int(targetrate * querynum)
        # 得到查中结果
        targetresult = []

        for searchresult in self.SearchResutlist:
            queryid = searchresult.QueryID

            if queryid.queryname.split('_')[0] in searchresult.SearchResultID:
                index = searchresult.SearchResultID.index(queryid.queryname.split('_')[0])

                sim = searchresult.SearchResultSimilarity[index]
                targetresult.append(sim)
        # 得到最小的阈值
        sim_sort = sorted(targetresult, reverse=True)

        if len(targetresult) == 0:
            print(" no target！！！")
            return 0
        threshold = sim_sort[targetnum - 1]  # Min Similarity as Threshold
        # print(threshold)
        for searchresult in self.SearchResutlist:
            # 返回的前n位累积匹配结果中相似度大于等于阈值T的所有结果数
            more_threshold_result = searchresult.SearchResultSimilarity[
                searchresult.SearchResultSimilarity >= threshold]
            np_SearchResultID = np.array(searchresult.SearchResultID)
            more_threshold_resultid = np_SearchResultID[searchresult.SearchResultSimilarity >= threshold]
            more_threshold_result_num += more_threshold_result.shape[0]
            queryid = searchresult.QueryID
            # 返回的前n位累积匹配结果中相似度大于等于阈值T的所有误识结果数
            more_threshold_falseresult_num += more_threshold_result.shape[0]
            if queryid.queryname.split('_')[0] in more_threshold_resultid:
                more_threshold_falseresult_num -= 1
        # print(threshold,more_threshold_falseresult_num,more_threshold_result_num)
        return round(float(more_threshold_falseresult_num) / float(more_threshold_result_num), 4), round(threshold, 4)

    def get_passunderfppi_match(self, fppirate, matchnum, topn):  # 找到阈值，返回大于该阈值且在top-K条件下的recall rate
        """
        计算确定误报率下的通过率
        :param fppirate: float
        :param topn :  int
        :param matchnum: 计算次数
        :return: 确定误报率下的通过率
        """
        # 得到误报次数
        falsematcnnum = int(fppirate * matchnum)
        if (falsematcnnum) > 0:
            falsematcnnum = falsematcnnum - 1
            # 得到错误匹配结果
        falsematchresult = []
        # 前n位累积匹配结果中相似度大于等于阈值T的所有结果数
        more_threshold_result_num = 0
        # 前n位累积匹配结果中相似度大于等于阈值T的所有查中结果数
        more_threshold_targetresult_num = 0

        for searchresult in self.SearchResutlist:
            SearchResultID = searchresult.SearchResultID[:topn]
            SearchResultSimilarity = searchresult.SearchResultSimilarity[:topn]
            queryid = searchresult.QueryID
            for i, resultid in enumerate(SearchResultID):
                if queryid.queryname.split('_')[0] != resultid:
                    sim = SearchResultSimilarity[i]
                    falsematchresult.append(sim)
        # 得到最小的阈值
        sim_sort = sorted(falsematchresult, reverse=True)
        if len(sim_sort) == 0:
            print("no falsematch!!")
            return 0
        assert len(sim_sort) > falsematcnnum
        threshold = sim_sort[falsematcnnum]

        for searchresult in self.SearchResutlist:
            SearchResultSimilarity = searchresult.SearchResultSimilarity[:topn]
            SearchResultID = searchresult.SearchResultID[:topn]
            # 返回的前n位累积匹配结果中相似度大于等于阈值T的所有结果数
            more_threshold_result = SearchResultSimilarity[SearchResultSimilarity >= threshold]
            np_SearchResultID = np.array(SearchResultID)
            more_threshold_resultid = np_SearchResultID[SearchResultSimilarity >= threshold]
            more_threshold_result_num += more_threshold_result.shape[0]
            queryid = searchresult.QueryID

            # 返回的前n位累积匹配结果中相似度大于等于阈值T的所有查中结果数
            if queryid.queryname.split('_')[0] in more_threshold_resultid:
                more_threshold_targetresult_num += 1

            # print(more_threshold_targetresult_num,more_threshold_result_num)

        return round(float(more_threshold_targetresult_num) / float(more_threshold_result_num), 4), round(threshold, 4)

    def get_passunderfppi_1vs1(self, fppirate):  # 以排在第fppirate的负对作为阈值，计算precision，TP/(TP+FP)
        """
        确定误识率下的通过率
        :param fppirate:误识率
        :return:
        """

        # 获得总比对次数
        pairs = len(self.SearchResutlist)
        sim_list = []
        flag_list = []
        for result in self.SearchResutlist:
            sim = float(result.SearchResultSimilarity)
            sim_list.append(sim)
            flag = result.TestType
            flag_list.append(flag)
        sim_list = np.array(sim_list)
        flag_list = np.array(flag_list)
        # cal
        neg_flag_list = np.where(flag_list == '0')[0]
        thred = sim_list[neg_flag_list][int(pairs * fppirate)]
        # print(thred)
        num_call = len(np.where(sim_list >= thred)[0])
        pass_call = np.where(flag_list[np.where(sim_list >= thred)[0]] == '1')
        return round(len(pass_call[0]) / num_call, 4), round(thred, 4)

    def get_fppiundertarget_1vs1(self, targetrate):  # 以排在第target rate的正对得分作为阈值，计算precision TP/(TP+FP)
        # get similarity
        sim_list = []
        flag_list = []
        for result in self.SearchResutlist:
            sim = float(result.SearchResultSimilarity)
            sim_list.append(sim)
            flag = result.TestType
            flag_list.append(flag)

        sim_list = np.array(sim_list)
        flag_list = np.array(flag_list)

        # cal
        pos_flag_list = np.where(flag_list == '1')[0]
        thred = sim_list[pos_flag_list][int(pos_flag_list.shape[0] * targetrate)]
        num_call = len(np.where(sim_list >= thred)[0])
        wrong_call = np.where(flag_list[np.where(sim_list >= thred)[0]] == '0')
        # print(len(wrong_call[0]),num_call)
        return round(len(wrong_call[0]) / num_call, 4), round(thred, 4)













