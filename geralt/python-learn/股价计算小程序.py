"""
定义如下变量：
name，公司名
stock_price，当前股价
stock_code，股票代码
stock_price_daily_growth_factor，股票每日增长系数，浮点类型，比如1.2
growth_days，增长天数
计算，经过growth_days天的增长后，股价达到多少钱
使用字符串格式化进行输出，如果是浮点数，要求小数点精确到2位数
"""
name = "传智播客"
stock_code = "003032"
# 数字不能有0开头
stock_price = 19.99
stock_price_daily_growth_factor = 1.2
print(f"公司：{name}，股票代码：{stock_code}，当前股价：{stock_price}")
print("经过七天的增长后，股票达到了%.2f" % (stock_price * stock_price_daily_growth_factor ** 7))


