# coding=utf-8
import pandas as pd
import numpy as np
import statsmodels.api as sm
import Queue
import tushare as ts
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

"""
Author:   李子杰、董煜
Date:  2017.9.5
Version:  8.0

本程序按照广发证券《广发TD线:在趋势中把握波段》开发
程序包含了:  1.获取数据部分 2.策略部分 3.指标定义 4.回测部分
读取本地数据时注意修改接口内地址

"""
# ------------------------------------------------------------------------------------------
# ----------------------------------------获取行情数据-------------------------------------
# ------------------------------------------------------------------------------------------

# 财经数据包tushare,后续可以接入wind接口
def get_stock_data_online(stock_code):
    """
    :param stock_code: 股票代码
    :return: 返回股票数据集（日期,开盘价，收盘价，最高价,最低价,涨跌幅）
    """
    stock_data = ts.get_hist_data(stock_code)
    stock_data = stock_data[['open', 'close', 'high', 'low', 'p_change']]
    stock_data['change'] = stock_data['p_change']
    stock_data.index = pd.DatetimeIndex(stock_data.index)
    stock_data['date'] = stock_data.index
    del stock_data['p_change']
    stock_data = stock_data.sort(ascending=True)
    stock_data['index'] = np.array(np.linspace(0, len(stock_data) - 1, len(stock_data)))
    stock_data.set_index('index', inplace=True)
    return stock_data

# 本地数据
def get_stock_data_local(stock_code):
    """
    :param stock_code: 股票代码
    :return: 返回股票数据集（日期，开盘价，收盘价，最高价,最低价,涨跌幅）
    """
    # 此处为存放csv文件的本地路径，请自行改正地址
    stock_data = pd.read_csv('/Users/lzj/Desktop/SNInvestment/Proj1/' + str(stock_code) + '.csv', parse_dates=['date'])
    stock_data = stock_data[['date', 'open', 'close', 'high', 'low', 'change']]

    stock_data.sort_values(by='date', inplace=True)
    stock_data.reset_index(drop=True, inplace=True)
    return stock_data

# 选择时间范围
def select_date_range(stock_data, start_date=pd.to_datetime('20060101'), end_date = pd.to_datetime('20150101')):
    """
    :param stock_data:
    :param start_date:
    :return:
    """
    stock_data_temp = stock_data.ix[(stock_data['date'] >= start_date) & (stock_data['date'] <= end_date), :]
    stock_data_temp.reset_index(inplace=True, drop=True)
    return stock_data_temp

# ------------------------------------------------------------------------------------------
# -----------------------------------------策略部分---------------------------------------
# ------------------------------------------------------------------------------------------

#  策略主体,输出仓位postion,1代表满仓,0代表平仓
def GFTD(stock_data_ori, m, n, a):
    """
    :param stock_data: 股票数据集
    :param m: TD点定义中的m,代表前后纳入考虑的数据个数
    :param n: 用来进行最小二乘法拟合的TD点的个数
    :param a: LLT延迟线因子
    :return: 当天收盘时持有该股票的仓位等数据。

    """
    stock_data = stock_data_ori.copy()
    # 对最高价,最低价进行LLT变换
    newhigh = LLT(stock_data['high'].values, a)
    stock_data['high'] = np.array(newhigh)
    newlow = LLT(stock_data['low'].values, a)
    stock_data['low'] = np.array(newlow)

    # 判断是否是需求点和供应点,并且使用 highTD 和 lowTD 来标记
    stock_data.ix[stock_data['high'] == stock_data['high'].shift(-m).rolling(min_periods=(2 * m + 1), window=(2 * m + 1), center=False).max(), 'highTD'] = 1
    stock_data.ix[stock_data['low'] == stock_data['low'].shift(-m).rolling(min_periods=(2 * m + 1), window=(2 * m + 1), center=False).min(), 'lowTD'] = 1

    # 对最高价,最低价进行LLT变换
    # newhigh = LLT(stock_data['high'].values, a)
    # stock_data['hhigh'] = np.array(newhigh)
    # newlow = LLT(stock_data['low'].values, a)
    # stock_data['llow'] = np.array(newlow)
    #
    # # 判断是否是需求点和供应点,并且使用 highTD 和 lowTD 来标记
    # stock_data.ix[stock_data['hhigh'] == stock_data['hhigh'].shift(-m).rolling(min_periods=2 * m + 1, window=2 * m + 1,
    #                                                                            center=False).max(), 'highTD'] = 1
    # stock_data.ix[stock_data['llow'] == stock_data['llow'].shift(-m).rolling(min_periods=2 * m + 1, window=2 * m + 1,
    #                                                                          center=False).min(), 'lowTD'] = 1

    # 填充highLimit,lowLimit,及上下两条TD线
    stock_data.loc[:, 'highslope'] = np.nan
    stock_data.loc[:, 'highlimit'] = np.nan
    stock_data.loc[:, 'lowslope'] = np.nan
    stock_data.loc[:, 'lowlimit'] = np.nan
    limit_caculation(stock_data, 1, n, m)
    limit_caculation(stock_data, 2, n, m)

    # 出现买入信号
    stock_data.ix[get_signal(stock_data, 1), 'position'] = 1
    # 出现卖出信号
    stock_data.ix[get_signal(stock_data, 2), 'position'] = 0

    # 将仓位填充完整,无信号的点与先前最近的一个信号点仓位保持一致
    stock_data['position'].fillna(method='ffill', inplace=True)
    stock_data['position'].fillna(0, inplace=True)

    return stock_data[['date', 'open', 'close', 'high', 'low', 'change', 'position','highTD','highslope','highlimit','lowlimit','lowslope','lowTD']]

# 判断是否有买卖信号
def get_signal(df, type):
    """
    :param df:
    :param type: 1判断买入信号,2判断卖出信号
    :return: True OR False
    """
    if type == 1:
        # 上穿且斜率大于0
        return (df['close'].shift(1) > df['highlimit'].shift(1)) & (df['lowslope'].shift(1) > 0)

    elif type == 2:
        # 下穿且斜率小于0
        return (df['close'].shift(1) < df['lowlimit'].shift(1)) & (df['highslope'].shift(1) < 0)


# 计算供应线和需求线,每一个点都有一个供应线上界highlimit和需求线下界lowlimit
def limit_caculation(df, type, n, m):
    """
    :param df:
    :param type: 1代表供给线(上界),2代表需求线(下界)
    :param numusedforols: 用来最小二乘的TD点个数
    """
    count = 0  # 计数器
    TDset = Queue.Queue()  # 队列

    if type == 1:
        for i in range(len(df)):
            if df.ix[i, 'highTD'] == 1:
                if TDset._qsize() >= n:
                    for j in range(int(1)):
                        [df.ix[i + j, 'highlimit'], df.ix[i + j, 'highslope']] = OLS(TDset, count + j)
                    TDset.get()
                TDset.put([count, df.ix[i, 'high']])
            elif (TDset._qsize() == n) & (pd.isnull(df.ix[i, 'highlimit'])):
                df.ix[i, 'highlimit'], df.ix[i, 'highslope'] = OLS(TDset, count)
            else:
                count += 1
                continue
            count += 1

    else:
        for i in range(len(df)):
            if df.ix[i, 'lowTD'] == 1:
                if TDset._qsize() >= n:
                    for j in range(int(1)):
                        [df.ix[i + j, 'lowlimit'], df.ix[i + j, 'lowslope']] = OLS(TDset, count + j)
                    TDset.get()
                TDset.put([count, df.ix[i]['low']])
            elif (TDset._qsize() == n) & (pd.isnull(df.ix[i, 'lowlimit'])):
                df.ix[i, 'lowlimit'], df.ix[i, 'lowslope'] = OLS(TDset, count)
            else:
                count += 1
                continue
            count += 1

# 最小二乘法
def OLS(tdset,count):
    """
    :param tdset: 特定日前前面的最近的n个TD点组成的队列,包含了行数(count)和价格
    :param count: 当前目标的行数
    :return: [目标拟合结果,斜率]
    """
    x = []
    y = []
    for i in range(tdset._qsize()):
        temp = tdset.get()
        x.append(temp[0])
        y.append(temp[1])
        tdset.put(temp)

    xx = np.array(x)
    newxx = sm.add_constant(xx)
    yy = np.array(y)
    model = sm.OLS(yy, newxx)
    results = model.fit()
    param = results.params
    return [param[0] + param[1] * count, param[1]]


# LLT延时变换
def LLT(list, a):
    result = []
    # 前两期不能迭代
    result.append(list[0])
    result.append(list[1])
    # 从第三期开始迭代计算LLT
    for i in range(len(list)):
        if i > 1:
            temp = (a-a * a / 4) * list[i] + (a * a / 2) * list[i - 1]\
                - (a - 3 * a * a / 4) * list[i - 2] + 2 * (1 - a) * result[i - 1]\
                - pow((1 - a), 2) * result[i - 2]
            result.append(temp)
    return result

# ------------------------------------------------------------------------------------------
# ----------------------------------------指标定义---------------------------------------
# ------------------------------------------------------------------------------------------

# 根据每日仓位计算总资产的日收益率
def account(df):
    """
    :param df: 股票账户数据集
    :return: 返回账户资产的日收益率和日累计收益率的数据集
    """
    df.ix[0, 'capital_rtn'] = 0
    # 当加仓时,计算当天资金曲线涨幅capital_rtn.capital_rtn = 昨天的position在今天涨幅 + 今天开盘新买入的position在今天的涨幅
    df.ix[df['position'] > df['position'].shift(1), 'capital_rtn'] = (df['close'] / df['open'] - 1)  * (df['position'] - df['position'].shift(1)) + df['change'] * 0.01 * df[
        'position'].shift(1)
    # 当减仓时,计算当天资金曲线涨幅capital_rtn.capital_rtn = 今天开盘卖出的positipn在今天的涨幅 + 还剩的position在今天的涨幅
    df.ix[df['position'] < df['position'].shift(1), 'capital_rtn'] = (df['open'] / df['close'].shift(1) - 1) * (df['position'].shift(1) - df['position']) + df['change'] * 0.01 * df['position']
    # 当仓位不变时,当天的capital_rtn是当天的change * position
    df.ix[df['position'] == df['position'].shift(1), 'capital_rtn'] = df['change'] * 0.01 * df['position']

    return df

# 根据每次买入的结果,计算相关指标
def trade_describe(df):
    """
    :param df: 包含日期、仓位和总资产的数据集
    :return: 输出账户交易各项指标``
    """
    # 计算资金曲线
    df['capital'] = (df['capital_rtn'] + 1).cumprod()
    # 记录买入或者加仓时的日期和初始资产
    df.ix[df['position'] > df['position'].shift(1), 'start_date'] = df['date']
    df.ix[df['position'] > df['position'].shift(1), 'start_capital'] = df['capital'].shift(1)
    df.ix[df['position'] > df['position'].shift(1), 'start_stock'] = df['close'].shift(1)
    # 记录卖出时的日期和当天的资产
    df.ix[df['position'] < df['position'].shift(1), 'end_date'] = df['date']
    df.ix[df['position'] < df['position'].shift(1), 'end_capital'] = df['capital']
    df.ix[df['position'] < df['position'].shift(1), 'end_stock'] = df['close']

    # 将买卖当天的信息合并成一个dataframe
    df_temp = df.ix[df['start_date'].notnull() | df['end_date'].notnull(), :]
    templist = df_temp.index.values
    # 记录交易周期
    result = []
    for i in range(len(templist) / 2):
        result.append(templist[2 * i + 1] - templist[2 * i] + 1)
    df_temp.ix[:, 'end_date'] = df_temp.ix[:, 'end_date'].shift(-1)
    df_temp.ix[:, 'end_capital'] = df_temp.ix[:, 'end_capital'].shift(-1)
    df_temp.ix[:, 'end_stock'] = df_temp.ix[:, 'end_stock'].shift(-1)

    # 构建账户交易情况dataframe：'hold_time'持有天数，'trade_return'该次交易盈亏,'stock_return'同期股票涨跌幅
    trade = df_temp.ix[df_temp['end_date'].notnull(), ['start_date', 'start_capital', 'start_stock',
                                                       'end_date', 'end_capital', 'end_stock']]
    trade.reset_index(drop=True, inplace=True)
    trade['hold_time'] = (trade['end_date'] - trade['start_date']).dt.days
    trade['trade_return'] = trade['end_capital'] / trade['start_capital'] - 1
    trade['trade_return_capital'] = trade['end_capital'] - trade['start_capital']
    trade.ix[trade['trade_return'] > 0, 'trade_return_signal'] = 1  # 盈利signal为1
    trade.ix[trade['trade_return'] < 0, 'trade_return_signal'] = 0  # 亏损signal为0
    trade['stock_return'] = trade['end_stock'] / trade['start_stock'] - 1

    trade_num = len(trade)  # 计算交易次数
    average_holdtime = sum(result) / trade_num  # 计算平均交易周期
    # total_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / float(365)  # 实际总年数
    trade_return_cum = df.iloc[len(df) - 1]['capital'] / df.iloc[0]['capital'] - 1  # 累积收益率
    trade_return_annual = pow(df.iloc[len(df)-1]['capital']/ df.iloc[0]['capital'],250.0/len(df)) - 1  # 年化收益率
    # trade_return_annual = pow(trade_return_cum + 1, 1 / total_years) - 1  # 年化收益率
    average_benefit = trade['trade_return'].mean()  # 单次平均收益率
    trade_positive_num = trade['trade_return_signal'].sum()  # 计算盈利次数
    trade_negative_num = trade_num - trade_positive_num  # 计算亏损次数
    win_times_rate = trade_positive_num / float(trade_num)  # 判断正确率
    trade_benefit_average = trade.ix[trade['trade_return_signal'] == 1, 'trade_return'].mean()  # 计算平均盈利率
    trade_loss_average = abs(trade.ix[trade['trade_return_signal'] == 0, 'trade_return'].mean())  # 计算平均亏损率
    trade_benefit = trade.ix[trade['trade_return_signal'] == 1, 'trade_return_capital'].sum()  # 计算总盈利
    trade_loss = abs(trade.ix[trade['trade_return_signal'] == 0, 'trade_return_capital'].sum())  # 计算总亏损
    trade_rate = trade_benefit_average / trade_loss_average  # 计算盈亏比
    max_gain = trade['trade_return'].max()  # 计算单笔最大盈利率
    max_loss = trade['trade_return'].min()  # 计算单笔最大亏损率


    #  输出账户交易各项指标
    print '\n==============每笔交易收益率及同期股票涨跌幅==============='
    print trade[['start_date', 'end_date', 'trade_return', 'stock_return']]
    print '\n====================账户交易的各项指标====================='
    print '交易次数为：%d' % trade_num
    print '平均交易周期为：%d' % average_holdtime
    print '累积收益率为: %f%%' % (trade_return_cum * 100)
    print '年化收益率为: %f%%' % (trade_return_annual * 100)
    print '单次平均收益率为:%f%%' % (average_benefit * 100)
    print '判断正确率为%f%%' % (win_times_rate * 100)
    print '平均盈利率为：%f%% \n平均亏损率为：%f%%' % (trade_benefit_average * 100, trade_loss_average * 100)
    print '盈亏比为：%f' % trade_rate
    print '正确次数为：%d \n错误次数为：%d' % (trade_positive_num, trade_negative_num)
    print '单次最大盈利为：%f%%  \n单次最大亏损为：%f%%' % (max_gain * 100, max_loss * 100)
    return trade_return_cum

# ------------------------------------------------------------------------------------------
# ----------------------------------------回测部分---------------------------------------
# ------------------------------------------------------------------------------------------

print '''
欢迎来到GFTD策略回测系统
本策略采用了广发证券的《广发TD线:在趋势中把握波段》,进行了纯多头开发
请输入需要回测的指数:
hs300 代表 沪深300   hs300_training 代表 沪深300样本期
csi500 代表 中证500  csi500_training 代表 中证500样本期
000001 代表 上证指数   000001_training 代表 上证指数样本期
'''
# stock_code=raw_input('请输入需要回测的代码:')
stock_code_whole = 'hs300'
stock_code_training = 'hs300_training'
# count=0
# fig = plt.figure()
# for stock_code in [stock_code_whole,stock_code_training]:
#     # ----------------获取数据----------------
#
#     # 1. 本地数据
#     # 全体(20050104-20161230):    csi500: 中证500  hs300: 沪深300  000001: 上证综指  zxb:  中小板
#     # 样本内(20050104-20101230):  csi500_training: 中证500  hs300_training: 沪深300  000001_training: 上证综指  zxb_training: 中小板
#
#     stock_set = get_stock_data_local(str(stock_code))  # 可自行修改数据集
#
#     # 2. online数据(可选,目前接入tushare接口)
#     # stock_set=get_stock_data_online('hs300')
#
#     # -------------运行策略,得到仓位-----------
#     # 设定参数,GFTD(stock_data, m, n, a)
#     # m: 用来判断TD点两侧取点个数   n: 用来进行最小二乘法拟合的TD点的个数   a: LLT延迟因子
#     # 中证500(3,2,0.05) 沪深300(2,3,0.05) 上证综指(2,2,0.05)
#     stock_set_result = GFTD(stock_set, 2, 3, 0.05)
#
#     # ----------------计算指标----------------
#     account(stock_set_result)
#     trade_describe(stock_set_result)
#
#     # 画图
#     count+=1
#     ax1=fig.add_subplot('21'+str(count))
#     # ax1.plot(stock_set_result[['close','highlimit','lowlimit']])
#     ax1.plot(stock_set_result[['close','highlimit','lowlimit']])
#     ax2=ax1.twinx()
#     ax2.plot(stock_set_result['capital'],color='r')
#     ax2.set_ylim([0,25])
#     # plt.plot(stock_set_result[['close','position']])
#     #     count+=1
# plt.show()
# # 保存至本地文档
# # downloadfile = stock_set_result.ix[:,['date','close','low','lowlimit','lowslope','lowTD','high','highlimit','highslope','highTD','position']]
# # downloadfile.to_csv('/Users/lzj/Desktop/9803.csv',sheet_name='Shit')
# #
stock_set = get_stock_data_online('000033')
print stock_set