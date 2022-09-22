from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import Counter
from mengzi.prompts import create_input_with_prompt


class MengziZeroShot(object):
    def __init__(self):
        pass

    # def prompt_map(self, task_type, input_string, label_list=None):
    def load(self, model_name="Langboat/mengzi-t5-base-mt"):
        # load model
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name)

    def token_decode(self, s):
        return self.tokenizer.decode(s, skip_special_tokens=True)

    def pick_most_common(self, x: list) -> str:
        return Counter(x).most_common(1)[0][0]

    def inference(self, task_type, input_string, input_string2=None, entity1=None, entity2=None):
        # make input text
        input_text = create_input_with_prompt(
            task_type, input_string, input_string2, entity1, entity2)
        # tokenize
        encodings = self.tokenizer(
            input_text, max_length=512, padding=True, return_tensors="pt")
        # print('encodings', encodings)

        # model inference
        outputs = self.model.generate(
            encodings['input_ids'], attention_mask=encodings['attention_mask'], max_length=512, num_beams=1)
        # print('outputs', outputs)

        # decode model output
        dec_out = list(map(self.token_decode, outputs))

        # print("dec_out: ", dec_out)
        # return result to web
        return self.pick_most_common(dec_out)


mz = MengziZeroShot()
mz.load()

# 使用示例：
# 批量测试
inputs = [  # 实体抽取
    {
        'task_type': 'entity_extraction',
        'input_string': '导致泗水的砭石受到追捧，价格突然上涨。而泗水县文化市场综合执法局颜鲲表示，根据监控，'
    },
    # 语义相似度
    {
        'task_type': 'text_similarity',
        'input_string': '你好，我还款银行怎么更换',
        'input_string_2': '怎么更换绑定还款的卡',
    },
    # 金融关系抽取
    {
        'task_type': 'financial_relationship_extraction',
        'input_string': '为打消市场顾虑,工行两位洋股东——美国运通和安联集团昨晚做出承诺,近期不会减持工行H股。',
        'entity1': "工行",
        'entity2': "美国运通",
    },
    # 广告文案生成
    {
        'task_type': 'ad_generation',
        'input_string': '类型-裤，版型-宽松，风格-潮，风格-复古，风格-文艺，图案-复古，裤型-直筒裤，裤腰型-高腰，裤口-毛边'},
    # 医学领域意图分类
    {
        'task_type': 'medical_domain_intent_classifier',
        'input_string': '呼气试验阳性什么意思'},

    # 情感分类
    {
        'task_type': 'sentiment_classifier',
        'input_string': '房间很一般，小，且让人感觉脏，隔音效果差，能听到走廊的人讲话，走廊光线昏暗，旁边没有什么可吃'},
    # 评论对象抽取
    {
        'task_type': 'comment_object_extraction',
        'input_string': '灵水的水质清澈，建议带个浮潜装备，可以看清湖里的小鱼。'
    },
    # 新闻分类
    {
        'task_type': 'news_classifier',
        'input_string': '懒人适合种的果树：长得多、好打理，果子多得都得送邻居吃'
    },
]

for t in inputs:
    task_type = t['task_type']
    print(f'task_type:{task_type}')
    input_string = t['input_string']

    if task_type == "text_similarity":
        input_string_2 = t['input_string_2']
        res = mz.inference(
            task_type=task_type, input_string=input_string, input_string2=input_string_2)
        print(f'input_string1:{input_string}')
        print(f'input_string2:{input_string_2}')
    elif task_type == "financial_relationship_extraction":
        res = mz.inference(task_type=task_type, input_string=input_string,
                           entity1=t['entity1'], entity2=t['entity2'])
    else:
        res = mz.inference(task_type=task_type, input_string=input_string)
        print(f'input_string:{input_string}')

    print(f'result:{res}')
    print("————————————————————————————————————————————")