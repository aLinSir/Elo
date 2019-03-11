# Elo
注意：所有数据都是模拟和虚构的，并不是真正的客户数据

我需要哪些文件？
您至少需要train.csv和test.csv文件。这些包含card_id我们将用于训练和预测的s。

该historical_transactions.csv和new_merchant_transactions.csv文件包含有关每张卡的交易信息。 historical_transactions.csv包含任何提供merchant_id的每张卡的最多3个月的交易。 new_merchant_transactions.csv包含两个月内新商家（merchant_id此特定card_id尚未访问过的商家）的交易。

merchants.csv包含merchant_id数据集中表示的每个的聚合信息。

我应该期望数据格式是什么？
数据格式如下：

train.csv和test.csv包含card_ids和卡本身的信息 - 卡片处于活动状态的第一个月等等 .train.csv也包含target。

historical_transactions.csv和new_merchant_transactions.csv旨在与train.csv，test.csv和merchants.csv连接。它们包含有关每张卡的交易的信息，如上所述。

商家可以与交易集合加入以提供额外的商家级信息。

我在预测什么？
您正在预测test.csv和sample_submission.csv中表示的每个忠诚度分数。card_id

文件说明
train.csv - 训练集
test.csv - 测试集
sample_submission.csv - 格式正确的示例提交文件 - 包含card_id您预期要预测的所有内容。
historical_transactions.csv - 每个历史交易最多3个月card_id
merchants.csv - 有关merchant_id数据集中所有商家的其他信息。
new_merchant_transactions.csv -两个月的有价值的数据对每个card_id包含所有购买card_id在做merchant_id的是呈S 没有在历史数据访问。
数据字段
Data Dictionary.xlsx中提供了数据字段描述。
