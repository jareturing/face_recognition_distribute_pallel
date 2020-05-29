# -*- coding: utf-8 -*-

import time
import os
import shutil
import xlwt,xlrd
import mxnet as mx
import cv2
import numpy as np
import pickle
from io_defination import *

def initialize(save_file_path):
    if (not os.path.exists(save_file_path)):
        os.mkdir(save_file_path)
    else:
        shutil.rmtree(save_file_path)
        os.mkdir(save_file_path)
def create_result_excel(similarity_save_name,all_search_reuslt,zhengjian_num,huoti_num,gallerynum,topk=50):
    """
    从查询结果中生成记录表
    :param similarity_save_name: 记录表名字
    :param all_search_reuslt: 查询结果
    :param zhengjian_num: 证件照数量
    :param huoti_num: 活体照数量
    :param gallerynum: 底库数量
    :param topk: 保存前N位结果
    :return:
    """
    xls = xlwt.Workbook()
    sheet1 = xls.add_sheet("sheet1")

    label = ['Query', 'Type']
    for i in range(topk):
        label.append(str(i+1))
        label.append("Sim")
    for key,value in enumerate(label):
        sheet1.write(0,key,value)
    for index,result in enumerate(all_search_reuslt):
        sheet1.write(index+1,0,result.QueryID.queryname)
        sheet1.write(index+1,1,result.QueryID.flag)
        for i in range(topk):
            sheet1.write(index+1,i*2+2,result.SearchResultID[i])
            sheet1.write(index+1,i*2+3,str(result.SearchResultSimilarity[i]))
    lastcontent = [zhengjian_num,huoti_num,gallerynum]
    for ll in range(3):
        sheet1.write(index+2,ll,lastcontent[ll])
    xls.save(similarity_save_name)
    print('file simliraty save to {}'.format(similarity_save_name))
def create_result_excel_1vs1(similarity_save_name,all_search_reuslt):
    """
    将查询结果写到表格中
    :param similarity_save_name:表格名字
    :param all_search_reuslt: 查询结果
    :return:
    """
    xls = xlwt.Workbook()
    sheet1 = xls.add_sheet("sheet1")
    title =  ["query","target","pos_neg","simlarity"]
    for i in range(len(title)):
        sheet1.write(0,i,title[i])
    sim_list = []
    flag_list = []
    query_list = []
    search_list = []
    for result in all_search_reuslt:
        sim = float(result.SearchResultSimilarity)
        sim_list.append(sim)
        flag = result.TestType
        flag_list.append(flag)
        queryname = result.QueryID
        query_list.append(queryname)
        searchname = result.SearchResultID
        search_list.append(searchname)
    # sort
    sim_list = np.array(sim_list)
    sim_index = np.argsort(-sim_list)
    sim_list = sim_list[sim_index]
    flag_list = np.array(flag_list)
    flag_list = flag_list[sim_index]
    query_list = np.array(query_list)
    query_list = query_list[sim_index]
    search_list = np.array(search_list)
    search_list = search_list[sim_index]

    for index in range(len(all_search_reuslt)):

        sheet1.write(index+1, 0, query_list[index])
        sheet1.write(index+1, 1, search_list[index])
        sheet1.write(index+1, 2, flag_list[index])
        sheet1.write(index+1, 3, str(sim_list[index]))
    xls.save(similarity_save_name)

def read_result_fromexcel_1vs1(similarity_save_name):
    """
    从记录表中读取结果
    :param similarity_save_name:名字
    :return:all_search_reuslt :结果
    """
    book = xlrd.open_workbook(similarity_save_name)
    sheet = book.sheet_by_index(0)
    print('read file from {}'.format(similarity_save_name))
    nrows = sheet.nrows
    ncols = sheet.ncols
    all_search_reuslt = []
    for i in range(1,nrows):
        queryname = sheet.cell(i, 0).value
        searchname = sheet.cell(i, 1).value
        posorneg = sheet.cell(i, 2).value
        simlarity = sheet.cell(i, 3).value

        result = SearchResultCompare(SearchID=i - 1,
                                     QueryID=queryname,
                                     SearchResultID=searchname,
                                     SearchResultSimilarity=simlarity,
                                     Type=posorneg)
        all_search_reuslt.append(result)

    return all_search_reuslt


def read_result_fromexcel(similarity_save_name):
    """
    从记录表中读取结果
    :param similarity_save_name:名字
    :return:
    """
    book = xlrd.open_workbook(similarity_save_name)
    sheet = book.sheet_by_index(0)
    print('read file from {}'.format(similarity_save_name))
    nrows = sheet.nrows
    ncols = sheet.ncols
    all_search_reuslt = []
    for i in range(1,nrows-1):
        queryname = sheet.cell(i, 0).value
        flag = sheet.cell(i, 1).value
        img = QueryInput(queryname, flag)
        result = SearchResult()
        result.SearchID = i-1
        result.QueryID = img
        result.SearchResultID = []
        result.SearchResultSimilarity = np.zeros(int(ncols/2)-1,)
        for j in range(1,int(ncols/2)):
            result.SearchResultID.append(sheet.cell(i, j*2).value)
            result.SearchResultSimilarity[j-1]=float(sheet.cell(i, j*2+1).value)
        all_search_reuslt.append(result)
    zhengjian_num = sheet.cell(nrows-1, 0).value
    huoti_num = sheet.cell(nrows-1, 1).value
    querynum = zhengjian_num+huoti_num
    gallerynum = sheet.cell(nrows-1, 2).value

    return all_search_reuslt,zhengjian_num,huoti_num,querynum,gallerynum
def cal_score(x1,x2,x,w):
    print(x1,x2,x,type(w))
    return round((x2-x)/(x2-x1)*w,4)
def cal_score_rv(x1,x2,x,w,reverse=True):
    print(x1,x2,x,type(w))
    if reverse:
        return round((x2-x)/(x2-x1)*w,4) if x2>x else 0
    else:
        return round((x-x2)/(x1-x2)*w,4) if x>x2 else 0
def create_score_table_1vs1(similarity_save_name,score):
    xls = xlwt.Workbook()
    sheet1 = xls.add_sheet("sheet1")
    title =  ["评测指标","指标","阈值","基准值","分数"]
    for i in range(len(title)):
        sheet1.write(0,i,title[i])
    zhibiao = ["95%查中率下的误识率","98%查中率下的误识率","99.5%查中率下的误识率","千分之一误识率下的通过率"]
    jizhun = [0.0002,0.0012,0.013,0.95]
    max = [0,0,0,1]
    weight = [8,8,8,6]
    for i in range(4):
        sheet1.write(i+1,0,zhibiao[i])
        sheet1.write(i+1,1,str(score[i*2]))
        sheet1.write(i+1,2,str(score[i*2+1]))
        sheet1.write(i+1,3,str(jizhun[i]))
        # sheet1.write(i+1,4,str(cal_score(max[i],jizhun[i],score[i*2],weight[i])))
        if i==3:
            sheet1.write(i + 1, 4, str(cal_score_rv(max[i], jizhun[i], score[i * 2], weight[i],False)))
        else:
            sheet1.write(i+1,4,str(cal_score_rv(max[i],jizhun[i],score[i*2],weight[i])))
    xls.save(similarity_save_name)
def create_score_table_1vsN(similarity_save_name,score):

    xls = xlwt.Workbook()
    sheet1 = xls.add_sheet("sheet2")
    title =  ["评测指标","指标","阈值","基准值","分数"]
    for i in range(len(title)):
        sheet1.write(0,i,title[i])
    zhibiao = ["证件照首位匹配率","证件照前5位匹配率","证件照前10位匹配率","活体照首位匹配率","活体照前5位匹配率","活体照前10位匹配率",
               "80%查中率下的误识率","90%查中率下的误识率","95%查中率下的误识率","百万分之一误识率下的通过率"]
    jizhun = [0.92,0.95,0.96,0.80,0.87,0.89,0.26,0.62,0.79,0.88]
    max = [0.99,0.99,1,0.98,0.99,1,0.01,0.05,0.40,0.99]
    weight = [15,3,2,23,5,2,5,5,5,5]
    for i in range(10):
        sheet1.write(i+1,0,zhibiao[i])
        sheet1.write(i+1,1,str(score[i*2]))
        sheet1.write(i+1,2,str(score[i*2+1]))
        sheet1.write(i+1,3,str(jizhun[i]))
        if score[i*2]!="NA":
            # sheet1.write(i+1,4,str(cal_score(max[i],jizhun[i],score[i*2],weight[i])))
            if i in [6,7,8]:
                sheet1.write(i + 1, 4, str(cal_score_rv(max[i], jizhun[i], score[i * 2], weight[i])))
            else:
                sheet1.write(i + 1, 4, str(cal_score_rv(max[i], jizhun[i], score[i * 2], weight[i],False)))
        else:
            sheet1.write(i+1, 4, "NA")

    xls.save(similarity_save_name)




def add_feature(Galleryname,GalleryFeature,Queryname,QueryFeature):
    """
    将两个特征合并，两个文件名列表合并
    :param Galleryname: 底库名
    :param GalleryFeature: 底库特征
    :param Queryname: 查询集名
    :param QueryFeature: 查询集特征
    :return: 合并后的名字列表，合并后的特征
    """
    Galleryname = Galleryname + Queryname
    GalleryFeature = np.concatenate(GalleryFeature,QueryFeature,axis=0)
    return Galleryname,GalleryFeature
def readpkl(pklname):
    f = open(pklname,'rb')
    Queryname = pickle.load(f)
    QueryFeature = pickle.load(f)
    f.close()
    print(len(Queryname))
    return Queryname,QueryFeature

def function_sim(query_result):
    pos1 = np.where((query_result>=-1)&(query_result<0.1))
    pos2 = np.where((query_result>=0.1)&(query_result<0.4))
    pos3 = np.where((query_result>=0.4)&(query_result<0.45))
    pos4 = np.where((query_result>=0.45)&(query_result<=1.0))
    query_result[pos1] =query_result[pos1] *0.4 + 0.36
    query_result[pos2] =query_result[pos2] + 0.3
    query_result[pos3] =query_result[pos3] *2 -0.1
    query_result[pos4] =query_result[pos4] *0.4 +0.6
    return query_result
def mergequeryfearure(pkl1,pkl2):
    Queryname = []

    Queryname1, QueryFeature1 = readpkl(pklname)
    print(QueryFeature1.shape)
    QueryFature = np.zeros(QueryFeature1.shape)
    count = 0
    for i,name in enumerate(Queryname1):
        if name.split('/')[-2] == 'huoti':
            Queryname.append(name)
            QueryFature[count] = QueryFeature1[i]
            count+=1
    print(len(Queryname))
    Queryname2, QueryFeature2 = readpkl(pklname)
    print(QueryFeature2.shape)
    for i, name in enumerate(Queryname2):
        if name.split('/')[-2] == 'zhengjian':
            Queryname.append(name)
            QueryFature[count] = QueryFeature2[i]
            count += 1

    print(len(Queryname))
    return Queryname,QueryFature
def separa_feature(feature,namelist):
    query_idx_list  =[]
    target_idx_list =[]

    for idx,name in enumerate(namelist):
        dir = name.split("/")[-2]
        if dir == 'target':
            query_idx_list.append(idx)
        else:
            target_idx_list.append(idx)
    print(len(query_idx_list),len(target_idx_list))

def create_pairs(pairstxt,datadir):
    """
    从人脸对文档中生成查询集名字和人脸对列表
    :param pairstxt: txt位置
    :param datadir: 人脸文件夹
    :return:
    """
    query_imgs = []
    paris_list = []
    with open(pairstxt, "r") as f:
        for line in f.readlines():
            splits = line.strip("\n").split("\t")
            pair_value = splits[0]
            queryname = os.path.join(datadir, splits[1].split("_")[0], splits[1])
            galleryname = os.path.join(datadir, splits[2].split("_")[0], splits[2])
            query_imgs.append(queryname)
            query_imgs.append(galleryname)
            paris_list.append(pair_value)
    return query_imgs,paris_list
