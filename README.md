# Decision-Tree

版本： `julia 1.0`
依赖的包: `BenchmarkTools`, `CSV`, `Random`, `DataFrames`, `Plots`

API：
- 类型定义：
    - `Leaf`
    - `Node`
    - `showTree` : 将树序列化`print`出来
- 训练：
    - `build` : 建树
    - `findFunction_ID3` : ID3模型
    - `findFunction_C45` : C4.5模型
    - `findFunction_CART` : CART模型
    - `k_fold_cross_validation` : $k$折交叉认证
    - `getDict` : 将原数据标签映射为整数
    - `transformDataSet` : 将原数据转换为向量的列表
- 预测
    - `getLabel` : 对一个向量预测其标签

> 注：数据放在`/data/`里面，原有数据没有`head`，需要手动加上`head`，不然无法正确读取csv文件。
