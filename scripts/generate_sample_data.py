"""
生成示例数据用于演示搜索功能
"""

import json
import os
from pathlib import Path
import random

# 示例数据
SAMPLE_DATA = {
    "时政": [
        {
            "title": "2020年中国失业人口统计数据发布",
            "url": "https://www.gov.cn/stats/2020/unemployment",
            "content": "国家统计局发布2020年就业统计数据。受新冠疫情影响，2020年中国城镇调查失业率有所上升，全年平均为5.6%，较2019年上升0.2个百分点。其中，2020年2月城镇调查失业率达到6.2%的历史高点。全年城镇新增就业人员1186万人，完成全年目标的131.8%。政府采取多项稳就业措施，包括减免社保费用、发放失业补助金等。"
        },
        {
            "title": "国务院部署2020年稳就业保民生措施",
            "url": "https://www.gov.cn/zhengce/2020/employment",
            "content": "国务院常务会议研究部署稳就业保民生政策措施。会议强调要千方百计稳定和扩大就业，确保完成全年就业目标任务。针对失业人员，加大失业保险待遇发放力度，扩大失业保险保障范围。对于农民工就业问题，要创造更多就地就近就业岗位。"
        },
        {
            "title": "人社部回应2020年失业率数据",
            "url": "https://www.mohrss.gov.cn/2020/data",
            "content": "人力资源和社会保障部新闻发言人就2020年就业形势答记者问。据统计，2020年全国失业人员再就业人数达到546万人，就业困难人员就业人数为163万人。城镇登记失业率保持在4%左右的较低水平。"
        },
        {
            "title": "2020年各省就业统计报告汇总",
            "url": "https://www.stats.gov.cn/2020/province",
            "content": "全国各省市2020年就业和失业统计数据汇总。广东省城镇新增就业139.3万人，城镇登记失业率为2.55%；浙江省城镇新增就业108.9万人；江苏省就业形势总体稳定。东北三省就业压力相对较大，辽宁、吉林、黑龙江城镇调查失业率高于全国平均水平。"
        },
        {
            "title": "两会代表热议2020年就业问题",
            "url": "https://www.xinhua.com/lianghui/2020/employment",
            "content": "全国两会期间，多位代表委员就2020年就业和失业问题建言献策。有代表建议加大对中小企业支持力度，稳定就业基本盘；有委员提出要发展新就业形态，支持灵活就业。政府工作报告明确提出就业优先政策要全面强化。"
        },
    ],
    "经济": [
        {
            "title": "2020年中国GDP增长2.3%",
            "url": "https://www.stats.gov.cn/gdp/2020",
            "content": "国家统计局发布2020年国民经济运行情况。初步核算，全年国内生产总值1015986亿元，按可比价格计算，比上年增长2.3%。中国成为全球唯一实现正增长的主要经济体。分季度看，一季度同比下降6.8%，二季度增长3.2%，三季度增长4.9%，四季度增长6.5%。"
        },
        {
            "title": "央行报告：2020年货币政策回顾",
            "url": "https://www.pbc.gov.cn/2020/report",
            "content": "中国人民银行发布2020年货币政策执行报告。全年三次降准释放长期流动性约1.75万亿元，引导贷款市场报价利率(LPR)下行。全年人民币贷款增加19.63万亿元，社会融资规模增量为34.86万亿元。"
        },
        {
            "title": "2020年A股市场总结",
            "url": "https://finance.sina.com.cn/stock/2020",
            "content": "2020年A股市场呈现结构性牛市特征。上证综指全年上涨13.87%，深证成指上涨38.73%，创业板指上涨64.96%。两市总成交额达206.7万亿元，较2019年增长62%。北向资金全年净流入2089亿元。"
        },
    ],
    "社会": [
        {
            "title": "2020年新冠疫情对民生影响报告",
            "url": "https://www.xinhua.com/2020/covid-livelihood",
            "content": "疫情对居民生活产生深远影响。部分行业从业者面临失业风险，餐饮、旅游、娱乐等服务业受冲击最大。政府及时出台救助政策，发放消费券、临时救助金，保障困难群众基本生活。"
        },
        {
            "title": "2020年全国低保数据统计",
            "url": "https://www.mca.gov.cn/2020/dibao",
            "content": "民政部公布2020年社会救助工作数据。全国共有城乡低保对象4426万人，其中城市低保对象818万人，农村低保对象3608万人。全年累计支出低保资金1609亿元。"
        },
    ],
    "军事": [
        {
            "title": "解放军2020年重大演训活动",
            "url": "https://www.81.cn/2020/training",
            "content": "2020年，中国人民解放军组织实施多场重大演训活动。东部战区在台海周边组织海空联合演练，南部战区在南海举行大规模军事演习。全军深入推进实战化练兵，战斗力水平持续提升。"
        },
    ],
    "科技": [
        {
            "title": "2020年中国科技创新成就",
            "url": "https://www.most.gov.cn/2020/innovation",
            "content": "2020年中国科技创新取得重大突破。嫦娥五号成功完成月球采样返回任务，天问一号火星探测器成功发射。量子计算机九章实现量子优越性，北斗三号全球卫星导航系统建成开通。"
        },
    ],
}


def generate_sample_data():
    """生成示例数据"""
    base_dir = Path(__file__).parent.parent / "data" / "classified"
    
    for category, docs in SAMPLE_DATA.items():
        cat_dir = base_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        for i, doc in enumerate(docs):
            doc["category"] = category
            doc["id"] = f"{category}_{i+1:04d}"
            
            file_path = cat_dir / f"{doc['id']}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 生成: {file_path}")
    
    print(f"\n✅ 示例数据生成完成！")


if __name__ == "__main__":
    generate_sample_data()
