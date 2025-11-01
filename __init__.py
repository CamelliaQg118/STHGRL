from .utils import fix_seed, load_data
from .STHGRL_model import sthgrl
from .STHGRL_model_xiaohy import sthgrl_xiaohy
from .STHGRL_model_degs import sthgrl_degs
# from .GNNs import GAT, GCN
from .clustering import mclust_R, leiden, louvain

# from module import *
# 此时被导入模块若定义了__all__属性，则只有__all__内指定的属性、方法、类可被导入。
# 若没定义，则导入模块内的所有公有属性，方法和类 。
__all__ = [
    "fix_seed",
    "sthgrl",
    "sthgrl_degs",
    "sthgrl_xiaohy",
    "mclust_R"
]
#相当于在from SDER_model import * 实际上只有Sedr类被导入，若Sedr不写入all中则导入的是SDER_model中全部的类或者函数